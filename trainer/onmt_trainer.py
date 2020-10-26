import torch
import torch.nn as nn
import onmt
import traceback

from onmt.trainer import Trainer
from onmt.utils.logging import logger
from loss.loss import NMTLossCompute, build_loss_compute, LabelSmoothingLoss
from emulator import Emulator


class MyTrainer(Trainer):
    def __init__(self,
                 model,
                 teacher_model,
                 train_loss,
                 valid_loss,
                 optim,
                 trunc_size=0,
                 shard_size=32,
                 norm_method='sents',
                 accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1,
                 gpu_rank=1,
                 gpu_verbose_level=0,
                 report_manager=None,
                 with_align=False,
                 model_saver=None,
                 average_decay=0,
                 average_every=1,
                 model_dtype='fp32',
                 earlystopper=None,
                 dropout=[0.3],
                 dropout_steps=[0],
                 source_noise=None,
                 emulator: Emulator = None):

        self.teacher_model = teacher_model
        self.n_gpu = n_gpu
        self.emulator = emulator
        super().__init__(model,
                         train_loss,
                         valid_loss,
                         optim,
                         trunc_size=trunc_size,
                         shard_size=shard_size,
                         norm_method=norm_method,
                         accum_count=accum_count,
                         accum_steps=accum_steps,
                         n_gpu=n_gpu,
                         gpu_rank=gpu_rank,
                         gpu_verbose_level=gpu_verbose_level,
                         report_manager=report_manager,
                         with_align=with_align,
                         model_saver=model_saver,
                         average_decay=average_decay,
                         average_every=average_every,
                         model_dtype=model_dtype,
                         earlystopper=earlystopper,
                         dropout=dropout,
                         dropout_steps=dropout_steps,
                         source_noise=source_noise)

    def sort_within_batches(self, tgt: torch.Tensor, indices: torch.Tensor):
        temp = tgt.transpose(0, 1)
        temp = temp[list(indices.tolist())]
        return temp.transpose(0, 1)

    def train(self,
              train_iter,
              train_steps,
              sos_id,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):

        if valid_iter is None:
            logger.info('Start training loop without validation....')
        else:
            logger.info(
                'Start training loop and validate every {step} steps'.format(
                    step=valid_steps))

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()

        self._start_report_manager(start_time=total_stats.start_time)

        # self.sos_id = train_iter.fields['tgt'].fields[0][1].vocab.stoi['<s>']

        for i, (batches,
                normalization) in enumerate(self._accum_batches(train_iter)):
            step = self.optim.training_step

            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info('GPURANK %d: index: %d', self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d" %
                            (self.gpu_rank, i + 1, len(batches)))
            if self.n_gpu > 1:
                normalization = sum(
                    onmt.utils.distributed.all_gather_list(normalization))

            self._gradient_accumulation(batches, normalization, total_stats,
                                        report_stats, sos_id,
                                        self.teacher_model)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps, self.optim.learning_rate(), report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d' %
                                (self.gpu_rank, step))
                valid_stats = self.validate(valid_iter,
                                            moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d' %
                                (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step,
                                  valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                    and (save_checkpoint_steps != 0
                         and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" \
                    else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                    else (batch.src, None)
                tgt, _ = batch.tgt if isinstance(batch.tgt, tuple) \
                    else (batch.tgt, None)

                src, tgt , src_lengths = src.to(torch.device('cuda')), tgt.to(
                    torch.device('cuda')), src_lengths.to(torch.device('cuda'))

                # F-prop through the model.
                outputs, attns = valid_model(src,
                                             tgt,
                                             src_lengths,
                                             bptt=False,
                                             with_align=self.with_align,
                                             teacher_outputs=None)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data,
                                         self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation(self,
                               true_batches,
                               normalization,
                               total_stats,
                               report_stats,
                               sos_id,
                               teacher_model=None):

        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt[0].size(0)
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            batch = self.maybe_noise_source(batch)
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer, tgt_lengths = batch.tgt if isinstance(
                batch.tgt, tuple) else (batch.tgt, None)
            if tgt_lengths is not None:
                tgt_lengths, indices = torch.sort(tgt_lengths, descending=True)
                tgt_outer = self.sort_within_batches(tgt_outer, indices)

            # tgt[seq_len,batch_size,1]
            def generate_tgt(tgt):
                batch_size = tgt.size(1)
                _device = torch.device(
                    'cuda') if self.n_gpu != 0 else torch.device('cpu')
                var = torch.full((tgt.size(0), batch_size, 1),
                                 int(sos_id),
                                 dtype=torch.long,
                                 device=_device)
                return var

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j:j + trunc_size]

                true_tgt = generate_tgt(tgt_outer[j:j + trunc_size])

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                teacher_outputs = None
                if teacher_model is not None:
                    # teacher_outputs[10,4,512]
                    teacher_outputs, _ = self.teacher_model(
                        tgt,
                        true_tgt,
                        lengths=tgt_lengths,
                        bptt=bptt,
                        with_align=self.with_align)
                    # word sampling method need to reshape the target tensor
                    if self.emulator is not None:
                        result = self.emulator.sample_from_batch(batch)
                        temp = []
                        for ten in result['predictions']:
                            temp.append(ten[0])
                        true_tgt = torch.stack(
                            temp).view(-1, batch.batch_size, 1)

                outputs, attns = self.model(src,
                                            tgt,
                                            src_lengths,
                                            bptt=bptt,
                                            with_align=self.with_align,
                                            teacher_outputs=teacher_outputs)
                '''
                outputs, attns = self.model(src,
                                            tgt,
                                            src_lengths,
                                            bptt=bptt,
                                            with_align=self.with_align,
                                            )
                '''
                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size,
                        teacher_outputs=teacher_outputs)

                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [
                            p.grad.data for p in self.model.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)

        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                tgt, tgt_lengths = batch.tgt if isinstance(
                    batch.tgt, tuple) else (batch.tgt, None)
                num_tokens = tgt.ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization


def build_trainer(opt,
                  device_id,
                  model,
                  fields,
                  optim,
                  model_saver=None,
                  emulator=None,
                  teacher_model=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    # TODO change tgt_field to teacher target field

    tgt_field = dict(fields)["tgt"].base_field

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    fuck = len(tgt_field.vocab)

    if opt.label_smoothing > 0:
        criterion = LabelSmoothingLoss(opt.label_smoothing,
                                       len(tgt_field.vocab),
                                       ignore_index=padding_idx)
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
    validation_criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
    
    teacher_loss_gen = teacher_model.generator if teacher_model is not None else None
    train_loss = NMTLossCompute(model.generator,
                                criterion,
                                use_distillation_loss=False,
                                teacher_generator=teacher_loss_gen)
    valid_loss = NMTLossCompute(model.generator, validation_criterion)

    #train_loss = build_loss_compute(model,tgt_field,opt,train=True,teacher_model=teacher_model)
    #valid_loss = build_loss_compute(model,tgt_field,opt,train=True)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
        device = torch.device('cuda')
        train_loss.to(device)
        valid_loss.to(device)

    else:
        gpu_rank = 0
        n_gpu = 0

    gpu_verbose_level = opt.gpu_verbose_level
    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    source_noise = None
    if len(opt.src_noise) > 0:
        src_field = dict(fields)["src"].base_field
        corpus_id_field = dict(fields).get("corpus_id", None)
        if corpus_id_field is not None:
            ids_to_noise = corpus_id_field.numericalize(opt.data_to_noise)
        else:
            ids_to_noise = None
        source_noise = onmt.modules.source_noise.MultiNoise(
            opt.src_noise,
            opt.src_noise_prob,
            ids_to_noise=ids_to_noise,
            pad_idx=src_field.pad_token,
            end_of_sentence_mask=src_field.end_of_sentence_mask,
            word_start_mask=src_field.word_start_mask,
            device_id=device_id)

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = MyTrainer(model,
                        teacher_model,
                        train_loss,
                        valid_loss,
                        optim,
                        trunc_size,
                        shard_size,
                        norm_method,
                        accum_count,
                        accum_steps,
                        n_gpu,
                        gpu_rank,
                        gpu_verbose_level,
                        report_manager,
                        with_align=True if opt.lambda_align > 0 else False,
                        model_saver=model_saver if gpu_rank == 0 else None,
                        average_decay=average_decay,
                        average_every=average_every,
                        model_dtype=opt.model_dtype,
                        earlystopper=earlystopper,
                        dropout=dropout,
                        dropout_steps=dropout_steps,
                        source_noise=source_noise,
                        emulator=emulator)
    return trainer
