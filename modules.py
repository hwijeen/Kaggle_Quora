import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaEmbedding(nn.Module):
    def __init__(self, pretrained_dim, proj_dim, dropout=0.0, type='DME',
                 vectors=None, freeze=True, scale=True):
        super().__init__()
        self.num_emb = len(vectors)
        self.pretrained_dim = pretrained_dim
        self.proj_dim = proj_dim
        self.type = type
        self.scale = scale
        if self.type == 'DME':
            # TODO: non-linearity?
            self.attn = nn.Sequential(nn.Linear(proj_dim, 2), nn.Linear(2,1))
            #self.attn = nn.Linear(proj_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.ModuleList(nn.Embedding.from_pretrained(v)
                                        for v in vectors)
        for emb in self.embeddings:
            emb.weight.requires_grad = not freeze
        self.linears = nn.ModuleList(nn.Linear(pretrained_dim, proj_dim)
                                      for _ in range(self.num_emb))

    def forward(self, x):
        projected_embs = [emb(x) for emb in self.embeddings]
        projected_embs = torch.cat([emb.unsqueeze(2)
                                    for emb in projected_embs], dim=2)
        if self.type == 'DME':
            attn_weights = F.softmax(self.attn(projected_embs).squeeze(),
                                     dim=-1).unsqueeze(2)
            emb_meta = torch.matmul(attn_weights, projected_embs).squeeze(2)
        elif self.type == 'unweighted':
            emb_meta = torch.sum(projected_embs, dim=2) / self.num_emb

        if self.scale:
            emb_meta = emb_meta * math.sqrt(self.proj_dim)
        return self.dropout(emb_meta)

class Embeddings(nn.Module):
    def __init__(self, num_vocab, dim_word, vectors=None, freeze=False):
        super().__init__()
        if vectors is not None:
            self.lut = nn.Embedding.from_pretrained(vectors)
        else:
            self.lut = nn.Embedding(num_vocab, dim_word)
        self.lut.weight.requires_grad = not freeze
        self.dim_word = dim_word

    def forward(self, x):
        """
        x: (batch_size, batch_len)

        return (batch_size, batch_len, d_model)
        """
        return self.lut(x) * math.sqrt(self.dim_word)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_word, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim_word)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_word, 2).float() *
                             -(math.log(10000.0) / dim_word))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, batch_len, d_model)

        return (batch_size, batch_len, d_model)
        """
        #x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + x.new_tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def get_embedding(num_vocab, dim_word, vectors, freeze_emb, embedding_type,
                  dropout_emb, transformer=False):
    if transformer and vectors is not None:
        print('Using only 1 pretrained word vector')
        return nn.Sequential(
            Embeddings(num_vocab, dim_word, vectors[0], freeze=freeze_emb),
            PositionalEncoding(dim_word, dropout_emb)
        )

    if vectors is None:
        return nn.Sequential(nn.Embedding(num_vocab, dim_word, padding_idx=1),
                                  nn.Dropout(dropout_emb))
    elif embedding_type == 'pretrained':
        assert len(vectors) == 1, 'Use only 1 pretrained embedding'
        return nn.Sequential(nn.Embedding.from_pretrained(vectors[0]),
                                  nn.Dropout(dropout_emb))
    else: # 'unweight', 'DME'
        assert len(vectors) >= 2, 'Use more than 2 pretrained embeddings when using DME'
        return MetaEmbedding(300, 300, dropout=dropout_emb,
                                  type=embedding_type, vectors=vectors,
                                  freeze=freeze_emb, scale=False)


def get_classifier(dim_hidden, dim_fc, dropout_fc):
    classifier = nn.Sequential(
        nn.Linear(dim_hidden, dim_fc), nn.ReLU(), nn.Dropout(dropout_fc),
        nn.Linear(dim_fc, dim_fc), nn.ReLU(), nn.Dropout(dropout_fc),
        nn.Linear(dim_fc, 1))
    return classifier


class dot_attention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, batch_len, 1, hidden_size *2)
        x = x.unsqueeze(2)
        # x_t: (batch_size, batch_len, hidden_size * 2, 1)
        x_t = x.transpose(-1, -2)
        # scores: (batch_size, batch_len)
        scores = torch.matmul(x, x_t).squeeze() / math.sqrt(x.size(3))
        # p_attn: (batch_size, 1, batch_len)
        p_attn = self.dropout(F.softmax(scores, dim=-1)).unsqueeze(1)
        # return (batch_size, hidden_size * 2)
        return torch.matmul(p_attn, x.squeeze(2)).squeeze(1)


class additive_attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, batch_len, hidden_size * 2)
        # scores: (batch_size, batch_len)
        scores = self.linear(x).squeeze() / math.sqrt(x.size(2))
        # p_attn: (batch_size, 1, batch_len)
        p_attn = self.dropout(F.softmax(scores, dim=-1).unsqueeze(1))
        return torch.matmul(p_attn, x).squeeze(1)


def _multihead_dot_attention(query, dropout=None, mask=None):
    batch_size, h, _, d_k = query.size()
    # query: (batch_Size, h, batch_len, 1, d_k)
    # query_t: (batch_size, h, batch_len, d_k, 1)
    query = query.unsqueeze(3)
    query_t = query.transpose(-1, -2)
    # scores: (batch_size, h, batch_len, 1, 1)
    scores = torch.matmul(query, query_t) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        scores.masked_fill_(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-3)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # _encoded: (batch_size, h, batch_len, d_k
    _encoded = torch.matmul(p_attn, query).squeeze(3)
   # (batch_size, batch_len, h*d_k)
    _encoded = _encoded.transpose(1, 2).contiguous() \
        .view(batch_size, -1, h * d_k)
    return torch.sum(_encoded, 1)


#def _multihead_dot_attention(query, dropout=None):
#    # query: (batch_size, h, batch_len, d_k)
#    batch_size, h, _, d_k = query.size()
#    # query_t: (batch_size, h, d_k, batch_len)
#    query_t = query.transpose(-1, -2)
#    # socres: (batch_size, h, batch_len, batch_len)
#    scores = torch.matmul(query, query_t) / math.sqrt(d_k)
#    p_attn = F.softmax(scores, -1)
#    if dropout is not None:
#        p_attn = dropout(p_attn)
#    # out: (batch_size, h, batch_len, d_k)
#    _encoded = torch.matmul(p_attn, query).transpose(1, 2).contiguous()\
#            .view(batch_size, -1, h * d_k)
#    return torch.sum(_encoded, 1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.proj_layer = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.attn = None  # ?
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)  # batch_first!!
        x_proj = self.proj_layer(x).view(batch_size, -1, self.h, self.d_k)\
                 .transpose(1, 2)
        _encoded = _multihead_dot_attention(x_proj, dropout=self.dropout,
                                            mask=mask)
        return self.linear(_encoded)
        #return _encoded

def get_attention(attention_type, dim_hidden=None, dropout=0.0, h=4):
    if attention_type == 'dot':
        return dot_attention(dropout)
    elif attention_type == 'additive':
        return additive_attention(dim_hidden, dropout)
    elif attention_type == 'multihead':
        return MultiHeadedAttention(h, dim_hidden, dropout=dropout)
    return NotImplementedError, 'Unknown attention type'
