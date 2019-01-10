import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO make modules
from utils import param_init
from modules import get_embedding, get_classifier

class CNNModel(nn.Module):
    # TODO: attention?
    def __init__(self, embedding, cnns, classifier):
        super(CNNModel, self).__init__()
        self.embedding = embedding
        self.cnns = cnns
        self.classifier = classifier

    def forward(self, x):
        """
        x: (batch_size, batch_len)
        """
        # embed: (batch_size, 1, batch_len, dim_word)
        embed = self.embedding(x).unsqueeze(1)
        # convolved[0]: (batch_size, 100, _)
        convolved = [F.relu(conv(embed)).squeeze(3) for conv in self.cnns]
        # pooled[0] : (batch_size, 100)
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convolved]
        # encoded : (batch_size, 100 * len(cnns))
        encoded = torch.cat(pooled, dim=1)
        out = self.classifier(encoded).squeeze()
        return out

    # TODO: model base class!!
    def summary(self):
        print('========== MODEL SUMMARY ==========')
        print('name of model: {}'.format(self.__class__.__name__))
        num_params = sum([param.nelement() for param in self.parameters()])
        print('number of params: {:,}'.format(num_params))
        print('===================================')


def build_cnn_model(num_vocab, dim_word, dim_fc,
                    windows, dim_feature,
                    dropout_emb=0.0, dropout_fc=0.0,
                    embedding_type=None, vectors=None, freeze_emb=True,
                    device=None):
    embedding = get_embedding(num_vocab, dim_word, vectors, freeze_emb,
                              embedding_type, dropout_emb)
    cnns = nn.ModuleList(nn.Conv2d(1, dim_feature, (w, dim_word))
                         for w in windows)
    dim_hidden = len(cnns) * dim_feature
    classifier = get_classifier(dim_hidden, dim_fc, dropout_fc)
    model = CNNModel(embedding, cnns, classifier)
    param_init(model)
    return model.to(device)




