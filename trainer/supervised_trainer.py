from __future__ import division

import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch import optim

import model.seq2seq as seq2seq
from evaluator.evaluator import Evaluator
from loss.loss import NLLLoss
from optim.optimer import Optimizer
from util.checkpoint import Checkpoint


class SupervisedTrainer(object):
    def __init__(self, export_dir='experiment',
                 loss=NLLLoss(),
                 batch_size=64,
                 random_seed=None,
                 checkpoint_every=100,
                 print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(export_dir):
            export_dir = os.path.join(os.getcwd(), export_dir)
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable: torch.Tensor, input_lengths, target_variable, model, teacher_model=None, teacher_forcing_ratio=.5):
        loss = self.loss
        loss.reset()

        input_batch_size = input_variable.size(0)
        input_seq_length = input_variable.size(1)
        target_batch_size = target_variable.size(0)
        target_seq_length = target_variable.size(1)

        if teacher_model is not None:
            tgt_arr = np.array([[model.sos_id]]*batch_size, 'i')
            tgt_var = torch.autograd.Variable(
                torch.LongTensor(tgt_arr).type(torch.LongTensor))

            transformer_output = model(input_variable, tgt_var)
            # transformer_output [batch_size,1,tgt_vocab_size]
            teacher_output, teacher_hidden, other = teacher_model(
                target_variable, target_variable.size(1))

        else:
            input_var = input_variable.view(
                input_seq_length, input_batch_size, 1)
            target_var = target_variable.view(
                target_seq_length, target_batch_size, 1)
            aacc = input_lengths.size()
            decoder_output, other = model(
                input_var, target_var, input_lengths)

        for step, step_output in enumerate(decoder_output):

            if teacher_model is not None:
                # step_output =
                tgt = target_variable[:, step+1].contiguous().view(-1)
                #fuck = teacher_output[-step].permute(2,1,0)
                loss.eval_batch(step_output, tgt, teacher_output[-step])

            else:
                liner_func = nn.Linear(128, 8)
                dec_out = liner_func(step_output)
                
                dec_out: torch.Tensor = F.log_softmax(dec_out, dim=1)
                loss.eval_batch(dec_out.view(input_batch_size, -1),
                                target_variable[:, step + 1])

        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, teacher_model, n_epochs, start_epoch, start_step,
                       dev_data, teacher_forcing_ratio=0):
        log = self.logger
        print_loss_total = 0
        epoch_loss_total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(dataset=data, batch_size=self.batch_size,
                                                       sort=False, sort_within_batch=True,
                                                       sort_key=lambda x: len(
                                                           x.src),
                                                       device=device, repeat=False)

        step_per_epoch = len(batch_iterator)

        total_steps = step_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs+1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            for _ in range((epoch - 1) * step_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1
                input_var, input_length = getattr(batch, 'src')
                target_var = getattr(batch, 'tgt')

                loss = self._train_batch(
                    input_variable=input_var,
                    input_lengths=input_length,
                    target_variable=target_var,
                    model=model, teacher_model=teacher_model)

                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)
                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields['src'].vocab,
                               output_vocab=data.fields['tgt'].vocab).save(self.export_dir)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / \
                min(step_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (
                epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (
                    self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, teacher_model=None, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0):

        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(
                self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('param', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(
                model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step

        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(
                    model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" %
                         (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, teacher_model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
