import spacy
import torch
import torchtext
from torchtext.data import TabularDataset, Field, RawField,\
    BucketIterator
from itertools import chain
from math import sqrt

from logger import get_logger


PAD_WORD = '<pad>'
logger = get_logger('data_loading')


# TODO: where to put the two functions below?
# pad_idx is 1 all the time
def get_pad_idx(dataset):
    #assert field_name in list(dataset.fields.keys()), 'unknown field name'
    return dataset.fields['question'].vocab.stoi[PAD_WORD]


def zero_pad_vector(vectors, pad_idx):
    if vectors is None: return # no pretrained
    if not isinstance(vectors, list):
        vectors = list(vectors)
    for vector in vectors:
        torch.zero_(vector[pad_idx])


class MyRawField(RawField):
    def __init__(self):
        super(MyRawField, self).__init__()
        self.is_target = False


def get_tokenizer():
    spacy_en = spacy.load('en')
    return lambda text: [tok.text for tok in spacy_en.tokenizer(text)]


def get_char_tokenizer():
    spacy_en = spacy.load('en')
    def char_tokenizer(text):
        words = [tok.text for tok in spacy_en.tokenizer(text)]
        sent = chain.from_iterable(words)
        return list(sent)
    return char_tokenizer


class KaggleData(object):
    def __init__(self, train_path, test_path):
        logger.info(
            'train path: {}, test_path:{}'.format(train_path, test_path))
        self.train_path = train_path
        self.test_path = test_path
        self.pad_word = PAD_WORD

    def build_field(self, max_len, include_lengths=False):
        logger.info('building field with max_len of {}'.format(max_len))

        tokenizer = get_tokenizer()
        qid_field = MyRawField()
        question_field = Field(tokenize=tokenizer,
                            include_lengths=include_lengths,
                            pad_token=self.pad_word, batch_first=True,
                            preprocessing=lambda x: x[:max_len + 1])
        target_field = Field(sequential=False, use_vocab=False,
                            is_target=True, dtype=torch.float)

        self.fields = [('qid', qid_field), ('question', question_field),
                       ('target', target_field)]

    def build_vocab(self, name, max_vocab, min_freq, pretrained_vectors=None,
                    cache=None):
        scale = sqrt(1/300)
        self.fields[1][1].build_vocab(self.train_dataset, self.valid_dataset,
                                      self.test_dataset, min_freq=min_freq,
                                      max_size=max_vocab)
        self.vocab = self.train_dataset.fields[name].vocab

        if len(pretrained_vectors) >= 1:
            self.fields[1][1].vocab.load_vectors(
                                    [torchtext.vocab.Vectors(path,
                                        unk_init=lambda x: x.uniform_(-scale, scale),
                                        cache=cache)
                                    for path in pretrained_vectors])
            self.vocab.vectors = list(torch.chunk(self.vocab.vectors,
                                                  len(pretrained_vectors), dim=-1))

        self.pad_idx = get_pad_idx(self.train_dataset)
        zero_pad_vector(self.vocab.vectors, self.pad_idx)


    def build_dataset(self, split_ratio=0.9, stratified=True,
                      strata_field='target'):
        logger.info('building dataset... making Examples may take a while')
        entire_dataset = TabularDataset(self.train_path, format='csv',
                                        fields=self.fields, skip_header=True)
        self.train_dataset, self.valid_dataset = entire_dataset.split(
                                                    split_ratio=split_ratio,
                                                    stratified=stratified,
                                                    strata_field=strata_field)
        self.test_dataset = TabularDataset(self.test_path, format='csv',
                                                fields=self.fields[:2],
                                                skip_header=True)

    def build_iterator(self, batch_sizes, device=torch.device('cuda')):
        logger.info(
            'building iterators with batch sizes {}'.format(batch_sizes))
        self.train_iter, self.valid_iter, self.test_iterator = \
            BucketIterator.splits(
                    (self.train_dataset, self.valid_dataset, self.test_dataset),
                    batch_sizes=batch_sizes,
                    sort_key=lambda x: len(getattr(x, 'question')),
                    device=device, sort_within_batch=True)

    def summary(self):
        print('========== DATA SUMMARY ==========')
        self.data_ratio()
        self.true_ratio()
        print('number of vocabs: {:,}'.format(len(self.vocab)))
        print('==================================')

    def data_ratio(self):
        print('number of examples in train / dev / test: {} / {} / {}'.
              format(len(self.train_dataset), len(self.valid_dataset),
                     len(self.test_dataset)))

    def true_ratio(self):
        train_total = len(self.train_dataset)
        train_insincere = sum([1 for elem in self.train_dataset.examples
                               if elem.target == '1'])
        valid_total = len(self.valid_dataset)
        valid_insincere = sum([1 for elem in self.valid_dataset.examples
                               if elem.target == '1'])
        print('insincere question in train / valid: {:.2} / {:.2}'
              .format((train_insincere / train_total),
                      (valid_insincere / valid_total)))


def add_mask(batch, pad_idx):
    try:
        mask = (batch.question != pad_idx).unsqueeze(-2)
    except: # length included
        mask = (batch.question[0] != pad_idx).unsqueeze(-2)
    setattr(batch, 'mask', mask)
