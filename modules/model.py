"""

# Sparse Sequence-to-Sequence Models

model.py

"""

from activations_and_losses import *


import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from typing import List, Tuple, Dict, Set, Union



def get_activation_func(alpha):
    if alpha == 1.0:
        return F.softmax
    if alpha == 1.5:
        return entmax15_exact
    if alpha == 2.0:
        return entmax2_bisection
    raise ValueError("Choose alpha_attn among 1, 1.5 and 2")



class Seq2Seq(nn.Module):
    def __init__(self,
                 input_vocab,
                 output_vocab,
                 embed_size,
                 hidden_size,
                 device='cpu',
                 alpha_attn=1.0,
                 num_layers=2,
                 dropout_rate=0.3):
        super(Seq2Seq, self).__init__()

        self.device = device
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.input_size = len(input_vocab)
        self.output_size = len(output_vocab)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self._attn_matrix = Parameter(torch.Tensor(hidden_size, 2 * hidden_size))
        torch.nn.init.xavier_uniform_(self._attn_matrix)
        self._attn_matrix.to(self.device)

        # layers:

        self.src_embed = nn.Embedding(self.input_size, embed_size, padding_idx=0)
        self.tgt_embed = nn.Embedding(self.output_size, embed_size, padding_idx=0)

        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True)

        self.decoder_lstm_0 = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.decoder_lstm_1 = nn.LSTMCell(hidden_size, hidden_size)

        self.attn_product = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.attn_vec = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.out = nn.Linear(hidden_size, self.output_size)

        # special activation:
        self.activation_attn = get_activation_func(alpha_attn)


    def forward(self,
                src_raw: List[List[str]],
                tgt_raw: List[List[str]]) -> torch.Tensor:

        idx = list(range(len(src_raw)))
        idx = sorted(idx, key=lambda i: len(src_raw[i]), reverse=True)
        src_raw = src_raw[idx]
        tgt_raw = tgt_raw[idx]

        src = self.input_vocab.prepare_for_model(src_raw, device=self.device)
        tgt = self.output_vocab.prepare_for_model(tgt_raw, device=self.device)
        src_lens = [len(s) for s in src_raw]

        src_encodings, decoder_init_vec = self.encoder(src, src_lens)

        # (tgt_len - 1, batch_size, hidden_size)
        src_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float, device=self.device)
        for i, length in enumerate(src_lens):
            src_masks[i, length:] = 1
        attn_vecs = self.decoder(src_encodings, src_masks, decoder_init_vec, tgt[:-1])

        # (tgt_len - 1, batch_size, output_size)
        tgt_output = self.out(attn_vecs)

        return tgt_output.transpose(0, 1), tgt.transpose(0, 1)
    

    def encoder(self,
                src: torch.Tensor,
                src_lens: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        src_embed = self.src_embed(src)
        packed_src_embed = pack_padded_sequence(src_embed, src_lens)

        src_encodings, (last_hidden, last_cell) = self.encoder_lstm(packed_src_embed)

        src_encodings, _ = pad_packed_sequence(src_encodings)

        src_encodings = src_encodings.permute(1, 0, 2)

        # initialise hidden states in the decoder
        dec_init_cell_0 = last_cell[0]
        dec_init_cell_1 = last_cell[-2]
        dec_init_hidden_0 = torch.tanh(dec_init_cell_0)
        dec_init_hidden_1 = torch.tanh(dec_init_cell_1)

        return src_encodings, ((dec_init_hidden_0, dec_init_cell_0), (dec_init_hidden_1, dec_init_cell_1))
    

    def decoder(self,
                src_encodings: torch.Tensor,
                src_masks: torch.Tensor,
                decoder_init: Tuple[torch.Tensor, torch.Tensor],
                tgt: torch.Tensor) -> torch.Tensor:
        batch_size = src_encodings.size(0)

        attn_vec_t = torch.zeros(batch_size, self.hidden_size, device=self.device)

        tgt_embed = self.tgt_embed(tgt)

        decoder_hidden_0, decoder_hidden_1 = decoder_init

        attn_vecs = []

        for y_t in tgt_embed.split(split_size=1):
            y_t = y_t.squeeze(0)

            # input-feeding:
            x_t = torch.cat([y_t, attn_vec_t], dim=-1)

            h_t_0, c_t_0 = self.decoder_lstm_0(x_t, decoder_hidden_0)
            h_t_1, c_t_1 = self.decoder_lstm_1(h_t_0, decoder_hidden_1)

            # get attention weights
            intermediate = h_t_1.mm(self._attn_matrix).unsqueeze(1)
            attn_scores = intermediate.bmm(src_encodings.transpose(1, 2)).squeeze(1)

            attn_scores.data.masked_fill_(src_masks.byte(), -float('inf'))
            alignment_t = self.activation_attn(attn_scores)  # weights

            context_t = torch.bmm(alignment_t.unsqueeze(1), src_encodings).squeeze(1)

            attn_vec_t = torch.tanh(self.attn_vec(torch.cat([h_t_1, context_t], dim=1)))
            attn_vec_t = self.dropout(attn_vec_t)

            decoder_hidden_0 = (h_t_0, c_t_0)
            decoder_hidden_1 = (h_t_1, c_t_1)
            attn_vecs.append(attn_vec_t)

        attn_vecs = torch.stack(attn_vecs)
        
        return attn_vecs


