import torch
from onmt.translate.greedy_search import GreedySearch

from model.nmtmodel import NMTmodel


class Emulator(object):
    def __init__(
            self,
            model: NMTmodel,
            fields,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=1,
            random_sampling_topk=1,
            random_sampling_temp=1,
            ignore_when_blocking=frozenset(),
    ):
        self.model = model
        self.fields = fields

        tgt_field = dict(self.fields)['tgt'].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device(
            'cuda', self._gpu) if self._use_cuda else torch.device('cpu')

        self.n_best = n_best
        self.min_length = min_length
        self.max_length = max_length
        self.ratio = ratio
        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t]
            for t in self.ignore_when_blocking
        }

    def sample_from_batch(self, batch):
        with torch.no_grad():
            decode_strategy = GreedySearch(
                pad=self._tgt_pad_idx,
                bos=self._tgt_bos_idx,
                eos=self._tgt_eos_idx,
                batch_size=batch.batch_size,
                min_length=self.min_length,
                max_length=self.max_length,
                block_ngram_repeat=0,
                exclusion_tokens=self._exclusion_idxs,
                return_attention=False,
                sampling_temp=self.random_sampling_temp,
                keep_topk=self.sample_from_topk)
            return self._sample_batch_with_strategy(batch, decode_strategy)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, None)
        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        return src, enc_states, memory_bank, src_lengths

    def _decoder_and_generate(self,
                              decoder_in,
                              memory_bank,
                              batch,
                              memory_lengths,
                              src_map=None,
                              step=None,
                              batch_offset=None):
        dec_out, dec_attn = self.model.decoder(decoder_in,
                                               memory_bank,
                                               memory_lengths=memory_lengths,
                                               step=step)
        if 'std' in dec_attn:
            attn = dec_attn['std']
        else:
            attn = None
        log_probs = self.model.generator(dec_out.squeeze(0))

        return log_probs, attn

    def _sample_batch_with_strategy(self, batch,
                                    decode_strategy: GreedySearch):
        usr_src_map = False
        parallel_paths = decode_strategy.parallel_paths

        src, enc_states, memory_bank, src_lengths = self._run_encoder(
            batch=batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)
        results = {
            "scores": None,
            "predictions": None,
            "attention": None,
        }
        src_map = batch.src_map if usr_src_map else None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank, src_lengths, src_map)
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decoder_and_generate(
                decoder_input,
                memory_bank,
                batch,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset)

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(
                        x.index_select(1, select_indices) for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        #    results['log_probs'] = log_probs

        results['scores'] = decode_strategy.scores
        results['predictions'] = decode_strategy.predictions
        # predictions is a list which its len is same as batch_size
        results['attention'] = decode_strategy.attention
        return results

    @classmethod
    def from_opt(cls):
        pass
