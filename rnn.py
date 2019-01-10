import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import get_attention

class BiRNNEncoder(nn.Module):
    def __init__(self, rnn, mtype=None):
        super().__init__()
        self.rnn = rnn
        self.mtype = mtype

    # FIXME mask for compatiability... 어케 지울까
    def forward(self, embed, lengths=None, mask=None):
        assert lengths is not None, 'Lengths of input is needed when using RNN'
        lengths -= 1  # for index
        batch_size, batch_len, _ = embed.size()
        out, (hn, cn) = self.rnn(embed)

        if self.mtype == 'pool':
            encoded = self.pool(out, batch_size)
        elif self.mtype == 'avg':
            raise NotImplementedError
        elif self.mtype == 'concat':
            encoded = self.concat(out, batch_size, batch_len, lengths)
        return encoded

    def pool(self, out, batch_size):
        out = F.max_pool1d(out.transpose(1, 2), out.size(1)).squeeze(2)
        out = out.contiguous().view(batch_size, 2, -1)
        forward_pooled = out[:, 0]
        backward_pooled = out[:, 1]
        encoded = torch.cat([forward_pooled, backward_pooled], dim=1)
        return encoded

    def avg(self, out):
        pass

    def concat(self, out, batch_size, batch_len, lengths):
        out = out.view(batch_size, batch_len, 2, -1)
        forward = out[:, :, 0]
        h_0 = forward[:, 0]
        backward = out[:, :, 1]
        h_t = backward[range(batch_size), lengths]
        encoded = torch.cat([h_0, h_t], dim=1)
        return encoded


class AttnBiRNNEncoder(nn.Module):
    def __init__(self, rnn, attention=None):
        super().__init__()
        self.rnn = rnn
        assert attention is not None, 'Specify types of attention when using AttnBiRNNEncoder'
        self.attention = attention

    def forward(self, embed, lengths=None, mask=None):
        assert lengths is not None, 'Lengths of input is needed when using RNN'
        lengths -= 1  # for index
        batch_size, batch_len, _ = embed.size()
        out, _ = self.rnn(embed)

        out = out.view(batch_size, batch_len, 2, self.rnn.hidden_size)
        # out (batch_size, batch_len, hidden_size * 2)
        out = torch.cat([out[:, :, i] for i in range(out.size(2))], dim=2)
        encoded = self.attention(out, mask)
        return encoded


def get_rnn_encoder(dim_word, dim_hidden, num_layers, attention=None,
                    mtype=None, dropout_rnn=0.0, dropout_attn=0.0):
    LSTM = nn.LSTM(dim_word, dim_hidden, num_layers=num_layers,
                   dropout=dropout_rnn, bidirectional=True, batch_first=True)
    if attention:
        attention = get_attention(attention, dim_hidden * 2, dropout_attn)
        sent_encoder = AttnBiRNNEncoder(LSTM, attention)
    else:
        sent_encoder = BiRNNEncoder(LSTM, mtype)
    return sent_encoder


