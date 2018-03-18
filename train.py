#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import random
import logging
import sys
import numpy as np
import torch
from visdom import Visdom
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from data_load import load_data, BUCKETS, BatchData
from model import MIL_Model
from utils import AverageMeter, get_n_params, auc_para, auc_question, F1_para, F1_question



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='quasart/quasartsample.pkl', help='dataset path')
    parser.add_argument('--log_file', default='output.log',
                        help='path for log file.')
    parser.add_argument('--log_per_updates', type=int, default=100,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--epochs', type=int, default=40, help='train epochs')
    parser.add_argument('--batch_size',type=int, default=10, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='rnn hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='rnn stacked layers')
    parser.add_argument('--grad_clipping', type=float, default=10)
    parser.add_argument('--neg_margin', type=float, default=0.1)
    parser.add_argument('--pos_margin', type=float, default=0.1)
    parser.add_argument('--curve', type=str, default='0')
    parser.add_argument('--pos_neg_margin', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.1, help='only applied to SGD.')
    parser.add_argument('--momentum', type=float, default=0, help='only applied to SGD.')
    parser.add_argument('--optimizer', default='adamax', help='supported optimizer: adamax, sgd')
    parser.add_argument('--loss', default='merge', help='loss function')
    parser.add_argument('--model_name', default='last', help='loss function')
    return parser.parse_args()

args = parse_args()

 # setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def train():
    viz = Visdom(port=8093)
    win = viz.line(
            X=np.array([0]),
            Y=np.array([0]),
            name='ddd'
        )
    train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings = load_data(args.datafile)
    args.vocab_size = len(word2id)
    args.emb_dim = word_embeddings.shape[1]
    log.info('========== parameters =============')
    log.info(repr(vars(args)))

    log.info('===========bucket statistics===========')
    print('bid:%s %s %s %s' % ('bucket id', 'train', 'valid', 'test'))
    for bid in range(len(BUCKETS)):
        print('bid:%s %s %s %s' % (bid, len(train_tuple_b[bid][0]), len(valid_tuple_b[bid][0]), len(test_tuple_b[bid][0])))
    batchdata = BatchData(train_tuple_b, valid_tuple_b, test_tuple_b)

    mil_model = MIL_Model(args, embedding=word_embeddings, state_dict=None)

   # print(mil_model)
    #for idx, m in enumerate(mil_model.modules()):
    #   print(idx, '->', m)
    log.info('===========start train==========')
    bucket_sizes = [batchdata.get_sizes(i) for i in range(len(BUCKETS))]
    total_step = sum(bucket_sizes) // args.batch_size
    bids = []
    for i, bucket_size in enumerate(bucket_sizes):
        bids.extend( [i] *  (bucket_size//args.batch_size + 1) )
    random.shuffle(bids)
    print('total train step %s in one epoch' % total_step)
    print('total train %s samples' % sum(bucket_sizes))
    #ex = batchdata.get_train_batch(0, args.batch_size)
    for epoch in range(args.epochs):
        mil_model.network.train()
        for step in range(total_step):
            bid =  bids[step]
            ex = batchdata.get_train_batch(bid, args.batch_size)
            mil_model.update(ex)

            if step % args.log_per_updates == 0:
                log.info('epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}]'.format(
                    epoch, mil_model.updates, mil_model.train_loss.avg))

                viz.line(
                    X=np.array([epoch * total_step + step]),
                    Y=np.array([mil_model.train_loss.avg]),
                    win='loss',
                    name=args.curve,
                    update='append'
                    )
            # for debug
            #break
        # test model
        predictions = []
        gts = []
        nb_test = 0
        mil_model.network.eval()
        for bid in range(len(BUCKETS)):
            for ex in batchdata.get_test_batch(bid, args.test_batch_size, True):
                nb_test += len(ex[0])
                predictions.extend(mil_model.predict(ex))
                gts.extend(ex[3].tolist())

        log.debug(gts[:20])
        log.debug(predictions[:20])
        auc_roc, pr_roc = auc_para(predictions, gts)
        log.info('[para level]auc score ROC: %s PR:  %s' % auc_para(predictions, gts))
        auc_roc_q, pr_roc_q = auc_question(predictions, gts)
        viz.line(
                X=np.array([epoch + 1]),
                Y=np.array([[pr_roc, auc_roc, auc_roc_q, pr_roc_q]]),
                win=win,
                update='append'
                )
        log.info('[question level]auc score ROC: %s PR: %s' % auc_question(predictions, gts))
        log.info('[para level]F1 score %s' % F1_para(predictions, gts))
        log.info('[question level]F1 score %s' % F1_question(predictions, gts))
        # TODO save model
        #mil_model.save(os.path.join('models', args.model_name + '.mdl'), epoch=epoch)

if __name__ == "__main__":
    train()
