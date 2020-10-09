import torch
import torch.nn as nn


class NMTmodel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTmodel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, **kwargs):
        dec_in = tgt[:-1]
        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        with_align = kwargs['with_align']

        if kwargs['bptt'] is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        if kwargs['teacher_outputs'] is not None:
            dec_out, attns = self.decoder(
                dec_in,
                memory_bank,
                memory_lengths=lengths,
                with_align=with_align,
                teacher_outputs=kwargs['teacher_outputs'])
        else:
            dec_out, attns = self.decoder(dec_in,
                                          memory_bank,
                                          memory_lengths=lengths,
                                          with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)