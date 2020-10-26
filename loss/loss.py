from __future__ import print_function

import math

import numpy as np
import onmt
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules.sparse_activations import LogSparsemax
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.utils.loss import LossComputeBase


def build_loss_compute(model, tgt_field, opt, train=True, teacher_model=None):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
            "order to use --lambda_coverage != 0"

    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(len(tgt_field.vocab),
                                                   opt.copy_attn_force,
                                                   unk_index=unk_idx,
                                                   ignore_index=padding_idx)
    elif opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(opt.label_smoothing,
                                       len(tgt_field.vocab),
                                       ignore_index=padding_idx)
    elif isinstance(model.generator[-1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)
    loss_gen = model.generator[0] if use_raw_logits else model.generator
    teacher_loss_gen = teacher_model.generator[
        0] if use_raw_logits else teacher_model.generator if teacher_model is not None else None

    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            criterion,
            loss_gen,
            tgt_field.vocab,
            opt.copy_loss_by_seqlength,
            lambda_coverage=opt.lambda_coverage)
    else:
        compute = NMTLossCompute(criterion, loss_gen, False, teacher_loss_gen)

    compute.to(device)

    return compute


class MyLoss(LossComputeBase):
    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None,
                 teacher_outputs=None):

        if trunc_size is None:
            trunc_size = batch.tgt[0].size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch,
                                             output,
                                             trunc_range,
                                             attns,
                                             teacher_outputs=teacher_outputs)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return None, batch_stats


class NMTLossCompute(MyLoss):
    def __init__(self,
                 generator,
                 criterion,
                 use_distillation_loss=False,
                 teacher_generator=None):

        super(NMTLossCompute, self).__init__(criterion, generator)

        self.generator = generator
        self.use_distillation_loss = use_distillation_loss
        self.teacher_generator = teacher_generator
        self._criterion = criterion
        self._KLDiveLoss = nn.KLDivLoss(reduction='batchmean')

    def _make_shard_state(self,
                          batch,
                          output,
                          range_,
                          attns=None,
                          teacher_outputs=None):
        res_dict = {}
        res_dict['output'] = output
        if teacher_outputs is not None:
            self.use_distillation_loss = True
            res_dict['teacher_outputs'] = teacher_outputs
        tgt, tgt_lengths = batch.tgt if isinstance(
            batch.tgt, tuple) else (batch.tgt, None)
        tgt = tgt.to(torch.device('cuda'))
        res_dict['target'] = tgt[range_[0] + 1:range_[1], :, 0]
        if tgt_lengths is not None:
            res_dict['target_lengths'] = tgt_lengths
        res_dict["copy_attn"] = attns.get("copy")
        #res_dict["align"] = None if not copy_attn else batch.alignment[range_[0] + 1: range_[1]]
        res_dict["coverage"] = attns.get("coverage")

        return res_dict

    # output is new model outputs

    def _compute_loss(self, batch, output, target, **kwargs):
        scores = self.generator(self._bottle(output))
        scores_data = scores.data.clone()

        target = target.view(-1)
        target_data = target.data.clone()

        loss = self._criterion(scores, target)

        if self.use_distillation_loss:
            weight_teacher_loss = 0.7
            teacher_outputs = kwargs['teacher_outputs']
            scores_teacher = self.teacher_generator(
                self._bottle(teacher_outputs))
            prob_teacher = scores_teacher.exp().detach()
            prob_scores = scores.exp().detach()
            scores, prob_teacher = self._pad(scores, prob_teacher)
            # Here we use a temperature of 1..
            loss_distilled = self._Jensen_Shannon_divergence(
                prob_scores , prob_teacher)
            loss = (1-weight_teacher_loss)*loss + \
                weight_teacher_loss*loss_distilled
        loss_data = loss.data.clone()
        stats = self._stats(loss_data, scores_data, target_data)

        return loss, stats

    def _Jensen_Shannon_divergence(self, score: torch.Tensor,
                                   target: torch.Tensor):
        mean_output = 0.5 * (score + target)
        src = 0.5 * self._KLDiveLoss(score,mean_output)
        tgt = 0.5 * self._KLDiveLoss(target,mean_output)
        return src + tgt

    def _pad(self, scores, teacher_scores):

        if scores.size(1) > teacher_scores.size(1):
            # teacher_scores must be padded to the same size as scores
            pad_size = scores.size(1) - teacher_scores.size(1)
            if pad_size % 2 == 0:
                pad_range = (pad_size // 2, pad_size // 2)
            else:
                pad_range = (pad_size // 2, (pad_size // 2) + 1)
            teacher_scores = nn.functional.pad(teacher_scores, pad_range)
            return scores, teacher_scores

        elif teacher_scores.size(1) > scores.size(1):
            pad_size = teacher_scores.size(1) - scores.size(1)
            if pad_size % 2 == 0:
                pad_range = (pad_size // 2, pad_size // 2)
            else:
                pad_range = (pad_size // 2, (pad_size // 2) + 1)
            scores = nn.functional.pad(scores, pad_range)
            return scores, teacher_scores

        else:
            return scores, teacher_scores


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.
    Yields:
        Each yielded shard is a dict.
    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(
                    zip(torch.split(state[k], shard_size),
                        [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size, ), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')
