import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from args import Args
from logger import get_logger
from utils import build_optimizer_scheduler, run_epoch, evaluate,\
    EarlyStoppingCriterion, inference, write_to_csv
from data_loading import KaggleData
from model import build_model

# for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.bechmark = False
np.random.seed(0)


def main():
    args = Args()

    logger = get_logger('main')
    EXP_NAME = 'multihead 4 layer with mask'
    assert EXP_NAME is not None, '이거슨 무슨 실험이냐!!'
    print(EXP_NAME)

    kaggle = KaggleData(args.train_path, args.test_path)
    kaggle.build_field(args.max_len, include_lengths=args.lengths)
    kaggle.build_dataset(split_ratio=0.9, stratified=False, strata_field='target')
    kaggle.build_vocab('question', args.max_vocab, min_freq=args.min_freq,
                       pretrained_vectors=args.embedding, cache=args.cache)
    kaggle.build_iterator(batch_sizes=[args.batch_size] * 3, device=args.device)
    kaggle.summary()

    logger.info('building model...')
    model = build_model(kaggle, args)

    #TODO: hyperparam pos_wieght is to be tuned
    criterion = nn.BCEWithLogitsLoss(reduction='sum',
                                     pos_weight=torch.tensor([args.pos_weight],
                                                             device=args.device))
    optimizer, scheduler = build_optimizer_scheduler('Adam', lr=0.001,
                                                     parameters=model.parameters(),
                                                     factor=0.5,
                                                     patience=args.scheduler_patience,
                                                     verbose=True)
    logger.info('start training...')
    early_stopping = EarlyStoppingCriterion(patience=args.early_stop_patience)
    for epoch in range(args.epoch):
        loss = run_epoch(model, kaggle.train_iter, criterion, optimizer)
        f1_score, accuracy = evaluate(model, kaggle.valid_iter,
                                      threshold=args.threshold, vocab=kaggle.vocab,
                                      verbose=False)
        scheduler.step(f1_score)
        print('loss at epoch {}: {:.5}'.format(epoch + 1, loss))
        print('f1 score / accuracy on valid: {:.4} / {:.4}'
              .format(f1_score, accuracy))

        if early_stopping(epoch, f1_score):
            if early_stopping.is_improved:
                logger.info('best model achieved in this epoch')
                # TODO: path name!!
                torch.save(model.state_dict(), 'best_model.pt')
        else:
            logger.info('early stopping...')
            break
        print()

    logger.info('best model is from epoch {} (f1: {:.4})'
                .format(early_stopping.best_epoch, early_stopping.best_score))
    model.load_state_dict(torch.load('best_model.pt'))

    logger.info('selecting threshold...')
    best = 0
    best_threshold = 0
    for th in np.arange(0.2, 0.6, 0.05):
        # FIXME: verbose
        f1_score, accuracy = evaluate(model, kaggle.valid_iter,
                                      threshold=float(th), vocab=kaggle.vocab,
                                      verbose=False)
        if f1_score > best:
            best = f1_score
            best_threshold = th
    print('best f1_score with threshold {}: {:.4} '.format(best_threshold, float(best)))

    pred_total, qid_total = inference(model, kaggle.test_iterator, best_threshold)
    write_to_csv(pred_total, qid_total, path='submission.csv')


if __name__ == "__main__":
    main()

