import torch.nn as nn

from rnn import get_rnn_encoder
from transformer import get_transformer_encoder
from modules import get_embedding, get_classifier
from utils import param_init


class SentenceClassifier(nn.Module):
    def __init__(self, name, embedding, sent_encoder, classifier):
        super().__init__()
        self.name = name
        self.embedding = embedding
        self.sent_encoder = sent_encoder
        self.classifier = classifier
        param_init(self)

    def forward(self, x, lengths=None, mask=None):
            return self.classifier(self.sent_encoder(self.embedding(x),
                                                     lengths=lengths,
                                                     mask=mask)).squeeze()


def build_model(data, args):
    # embedding
    is_transformer = True if args.model == 'transformer' else False
    embedding = get_embedding(len(data.vocab), args.dim_word,
                              data.vocab.vectors, args.freeze_emb,
                              args.embedding_type, args.dropout_emb,
                              transformer=is_transformer)

    # sentence encoder
    if args.model == 'rnn':
        sent_encoder = get_rnn_encoder(args.dim_word, args.dim_hidden,
                                       args.num_layers,
                                       attention=args.attention,
                                       mtype=args.mtype,
                                       dropout_rnn=args.dropout_rnn,
                                       dropout_attn=args.dropout_attn)
        classifier = get_classifier(args.dim_hidden * 2, args.dim_fc,
                                    args.dropout_fc)
    #elif args.model == 'cnn':
    #    return build_cnn_model(len(data.vocab), args.dim_word, args.dim_fc,
    #                           args.windows, args.dim_feature,
    #                           dropout_emb=args.dropout_emb,
    #                           dropout_fc=args.droprout_fc,
    #                           embedding_type=args.embedding_type,
    #                           vectors=data.vocab.vectors,
    #                           freeze_emb=args.freeze_emb, device=args.device)
    elif args.model == 'transformer':
        sent_encoder = get_transformer_encoder(args.dim_model, args.h, args.N,
                                               args.dim_ff, args.attention,
                                               args.dropout_transformer,
                                               args.dropout_attn)
        classifier = get_classifier(args.dim_model, args.dim_fc,
                                    args.dropout_fc)

    return SentenceClassifier(args.model, embedding, sent_encoder, classifier)\
        .to(args.device)

