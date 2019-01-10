import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import get_attention
from utils import clones

class LayerNorm(nn.Module):  # preserves dimension
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        x: (batch_size, batch_len, d_model)
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x: (batch_size, batch_len, d_model)
        """
        return x + self.dropout(sublayer(self.norm(x)))


class PositionalFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, batch_len)
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        x: (batch_size, batch_len_src, d_model)
        mask: (batch_size, 1, batch_len_src)

        return (batch_size, batch_len_src, d_model)
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # lambda!!
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        x: (batch_size, batch_len_src, d_model)
        mask: (batch_size, 1, batch_len_src)

        return: (batch_size, batch_len_src, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # this is memory


class TransformerEncoder(nn.Module):
    def __init__(self, encoder, attention):
        super().__init__()
        self.encoder = encoder
        self.attention = attention

    def forward(self, embed, lengths=None, mask=None):
        # embed: (batch_size, batch_len, dim_word)
        return self.attention(self.encoder(embed, mask))


def get_transformer_encoder(dim_model, h, N, dim_ff, attention,
                            dropout_transformer, dropout_attn):
    attn = get_attention('multihead', dim_hidden=dim_model,
                         dropout=dropout_transformer, h=h)
    ff = PositionalFeedForward(dim_model, dim_ff, dropout_transformer)
    encoder = Encoder(EncoderLayer(dim_model, attn, ff, dropout_transformer), N)
    attention = get_attention(attention, dim_model, dropout_attn)
    sent_encoder = TransformerEncoder(encoder, attention)
    return sent_encoder

