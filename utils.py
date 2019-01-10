import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from data_loading import PAD_WORD, add_mask


class EarlyStoppingCriterion(object):
    # TODO: min_delta
    def __init__(self, patience):
        self.patience = patience
        self.count = 0
        self.best_score = 0.0
        self.best_epoch = None

    def __call__(self, epoch, current_score):
        self.is_improved = current_score >= self.best_score
        if self.is_improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.count = 0
            return True
        else:
            self.count += 1
            return self.count < self.patience


def build_optimizer_scheduler(opt_name, lr, parameters, factor, patience,
                              verbose):
    if opt_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr)
    elif opt_name == 'RMSProp':
        optimizer = optim.RMSprop(parameters, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=factor,
                                                     patience=patience,
                                                     verbose=verbose)
    return optimizer, scheduler


def param_init(model):
    for name, p in model.named_parameters():
        if 'bias' in name:
            nn.init.uniform_(p, a=0, b=1) # small positive numbers
        elif p.dim() > 1 and 'embedding' not in name:
            #nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(p)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_pred(model, batch):
    add_mask(batch, 1)  # pad_idx is 1
    try:
        return model(batch.question, mask=batch.mask)
    except: # length included
        return model(*batch.question, mask=batch.mask)


def run_epoch(model, train_iter, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for train_batch in tqdm(train_iter, unit='sent'):
        pred = get_pred(model, train_batch)
        trg = train_batch.target  # (32,)
        loss = criterion(pred, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / train_iter.batch_size
    return epoch_loss / train_iter.iterations


def evaluate(model, valid_iter, threshold, vocab, verbose):
    model.eval()
    match = 0
    total = 0
    pred_total = np.empty(len(valid_iter.dataset))
    target_total = np.empty(len(valid_iter.dataset))
    index = 0
    for valid_batch in valid_iter:
        pred = get_pred(model, valid_batch)
        pred = (torch.sigmoid(pred) > threshold).cpu().numpy()
        pred_total[index: index + pred.shape[0]] = pred
        target = valid_batch.target.cpu().numpy()
        target_total[index: index + target.shape[0]] = target
        index += pred.shape[0]
        match += sum(pred == target)
        total += len(target)
    f1_score = metrics.f1_score(target_total, pred_total)
    # precision = metrics.precision_score(target_total, pred_total)
    # recall = metrics.recall_score(target_total, pred_total)
    if verbose:
        print_prediction(valid_batch, pred, vocab.itos)
        # print('precision: {}, recall: {}'.format(precision, recall))
    return f1_score, match / total


def inference(model, test_iter, threshold):
    qid_total = []
    pred_total = np.empty(len(test_iter.dataset))
    index = 0
    for test_batch in test_iter:
        pred = get_pred(model, test_batch)
        pred = (torch.sigmoid(pred) > threshold).cpu().numpy()
        pred_total[index: index + pred.shape[0]] = pred
        index += pred.shape[0]
        qid_total.extend(test_batch.qid)
    return pred_total, qid_total


def write_to_csv(pred_total, qid_total, path='submission.csv'):
    with open(path, 'w') as f:
        f.write('qid,prediction\n')
        for pred, qid in zip(pred_total, qid_total):
            f.write('{},{}\n'.format(qid, int(pred)))
    print('submission file saved at {}'.format(path))


def print_prediction(valid_batch, pred, vocab_list, upto=10):
    pred_insincere_idx = np.where(pred == 1)[0].tolist()
    pred_insincere = valid_batch.question[pred_insincere_idx]  # tensor
    answers = valid_batch.target[pred_insincere_idx][:upto]
    sentences = reverse(pred_insincere, vocab_list, upto)
    try:
        for sent, ans in zip(sentences, answers):
            print('({}), {}'.format(int(ans.item()), sent))
    except:
        return


def reverse(batch_tensor, vocab_list, upto=10):
    sentences = []
    for idx, example in enumerate(batch_tensor):
        sent = [vocab_list[idx.item()] for idx in example]
        sent = [word for word in sent if word != PAD_WORD]
        sentences.append(' '.join(sent))
        if idx == upto:
            return sentences


