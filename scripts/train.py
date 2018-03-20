#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.getcwd())
print(os.getcwd())
import os
import argparse
import random
import logging
import numpy as np
import torch
from visdom import Visdom
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from MIL.model import MIL_Model
from data.metrics import AverageMeter, get_n_params, auc_para, auc_question, F1_para, F1_question
from data import utils, vector, data, visual_helper

def parse_args():
    parser = argparse.ArgumentParser()
    # runtime environment
    parser.add_argument('--num-epochs', type=int, default=40,
                        help='Train data iterations')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--data-workers', type=int, default=5,
                        help='Batch size for training')
    parser.add_argument('--display-iter', type=int, default=25,
                        help='print interval')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='Batch size during validation/testing')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')

    # files
    parser.add_argument('--data-dir', default='quasart', help='data dir')
    parser.add_argument('--task', default='quasart', help='identity for plot')
    parser.add_argument('--train-file', default='sample_train-processed-spacy.txt', help='train data file')
    parser.add_argument('--dev-file', default='sample_val-processed-spacy.txt', help='train data file')
    parser.add_argument('--embed-dir', type=str, default='embeddings',
                        help='Directory of pre-trained embedding files')
    parser.add_argument('--embedding-file', type=str,
                        default='glove.840B.300d.txt',
                        help='Space-separated pretrained embeddings file')
    parser.add_argument('--log-file', default='output.log', help='log file')
    parser.add_argument('--sent-num', type=int, default=10, help='number of sentences')
    # save  load

    # model specific details
    parser.add_argument('--restrict-vocab', type=bool, default=False, help='restrict vocab to the embedding files')
    parser.add_argument('--hidden-size', type=int, default=128, help='rnn hidden size')
    parser.add_argument('--num-layers', type=int, default=1, help='rnn stacked layers')
    parser.add_argument('--grad-clipping', type=float, default=10)
    parser.add_argument('--neg-margin', type=float, default=0.1)
    parser.add_argument('--pos-margin', type=float, default=0.1)
    parser.add_argument('--curve', type=str, default='0')
    parser.add_argument('--pos-neg-margin', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=0.1, help='only applied to SGD.')
    parser.add_argument('--momentum', type=float, default=0, help='only applied to SGD.')
    parser.add_argument('--optimizer', default='adamax', help='supported optimizer: adamax, sgd')
    parser.add_argument('--loss', default='merge', help='loss function')
    parser.add_argument('--model-name', default=None, type=str, help='save model name')
    return parser.parse_args()

args = parse_args()

args.train_file = os.path.join(args.data_dir, args.train_file)
args.dev_file = os.path.join(args.data_dir, args.dev_file)
args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
if not args.model_name:
    import uuid
    import time
    args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

 # setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s  [%(filename)s,%(lineno)d-%(funcName)s]', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def init_from_scratch(args, train_exs, dev_exs):
    word_dict = utils.build_word_dict(args, train_exs + dev_exs)
    model = MIL_Model(args, word_dict, state_dict=None)
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)
    return model


def evaluate(args, model, data_loader, epoch, name):
    losses = utils.AverageMeter()
    pred_scores = []
    gts = []
    for idx, ex in enumerate(data_loader):
        score, target, loss = model.predict(ex)
        pred_scores.extend(score)
        gts.extend(target)
        losses.update(loss)
    auc_para_roc, auc_para_pr = auc_para(pred_scores, gts)
    auc_q_roc, auc_q_pr = auc_question(pred_scores, gts)

    viz.line(
            X=np.array([epoch]),
            Y=np.array([losses.avg]),
            win='%s-loss' % args.task,
            name=name,
            update='append'
            )
    return {'%s_auc_q_roc' % name: auc_q_roc, '%s_auc_q_pr' % name: auc_q_pr, '%s_auc_para_roc' % name:auc_para_roc, '%s_auc_para_pr' % name:auc_para_pr}



viz = Visdom(port=8093)
def train():
    viz.line(
            X=np.array([0]),
            Y=np.array([0]),
            win='%s-loss' % args.task,
            name='train',
            opts={'title': '%s-loss' % args.task}
        )

    viz.line(
            X=np.array([0]),
            Y=np.array([0]),
            win='%s-loss' % args.task,
            name='dev',
            update='append'
        )
    visual = visual_helper.Visual(win='%s-auc_metrics' % args.task, lines=['dev_auc_q_roc','dev_auc_q_pr','dev_auc_para_roc','dev_auc_para_pr',\
            'train_auc_q_roc','train_auc_q_pr','train_auc_para_roc','train_auc_para_pr'])
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args, args.train_file)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))
    logger.info('-' * 100)
    logger.info('Training model from scratch...')
    model = init_from_scratch(args, train_exs, dev_exs)
    model.network.cuda()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev.
    train_dataset = data.ReaderDataset(train_exs, model)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
    )
    dev_dataset = data.ReaderDataset(dev_exs, model)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
    )
    start_epoch = 0
    logger.info('-' * 100)
    logger.info('Starting training...')
    global_stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        global_stats['epoch'] = epoch
        train_loss = utils.AverageMeter()
        epoch_time = utils.Timer()
        for idx, ex in enumerate(train_loader):
            train_loss.update( model.update(ex) )
            if idx % args.display_iter == 0:
                logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(train_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
                train_loss.reset()
        results = evaluate(args, model, dev_loader, epoch, 'dev')
        results.update( evaluate(args, model, train_loader, epoch, 'train'))
        logger.info(results)
        visual.plot(results)


if __name__ == "__main__":
    train()
