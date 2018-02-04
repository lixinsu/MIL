#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from data_load import load_data, BUCKETS, BatchData
from model import MIL_Model
from utils import AverageMeter, get_n_params, auc_para



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='output.log',
                        help='path for log file.')
    parser.add_argument('--log_per_updates', type=int, default=100,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--epochs', default=20, help='train epochs')
    parser.add_argument('--batch_size', default=10, help='batch size')
    parser.add_argument('--test_batch_size', default=10, help='batch size')
    parser.add_argument('--hidden_size', default=128, help='rnn hidden size')
    parser.add_argument('--num_layers', default=1, help='rnn stacked layers')
    parser.add_argument('--grad_clipping', type=float, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.1, help='only applied to SGD.')
    parser.add_argument('--momentum', type=float, default=0, help='only applied to SGD.')
    parser.add_argument('-op', '--optimizer', default='adamax', help='supported optimizer: adamax, sgd')
    return parser.parse_args()


args = parse_args()
# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def train():
    train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings = load_data('quasart/quasart.pkl')
    args.vocab_size = len(word2id)
    args.emb_dim = word_embeddings.shape[1]
    log.info('========== parameters =============')
    log.info(repr(vars(args)))

    log.info('===========bucket statistics===========')
    print('bid:%s %s %s %s' % ('bucket id', 'train', 'valid', 'test'))
    for bid in range(len(BUCKETS)):
        print('bid:%s %s %s %s' % (bid, len(train_tuple_b[bid][0]), len(valid_tuple_b[bid][0]), len(test_tuple_b[bid][0])))
    batchdata = BatchData(train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings)

    mil_model = MIL_Model(args, embedding=word_embeddings, state_dict=None)

    log.info('===========start train==========')
    for epoch in range(args.epochs):
        for step in range(100):
            ex = batchdata.get_train_batch(0, args.batch_size)
            mil_model.update(ex)
            if step % args.log_per_updates == 0:
                log.info('epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}]'.format(
                    epoch, mil_model.updates, mil_model.train_loss.avg))

        # test model
        predictions = []
        gts = []
        nb_test = 0
        for ex in batchdata.get_test_batch(0, args.test_batch_size, True):
            nb_test += len(ex[0])
            predictions.extend(mil_model.predict(ex))
            gts.extend(ex[3].tolist())
        print(len(predictions))
        print(gts[:2])
        print(predictions[:2])
        auc_score = auc_para(predictions, gts)
        print('test auc score %s' % auc_score)
        # TODO save model
        mil_model.save('tmp.mdl', epoch=epoch)

if __name__ == "__main__":
    train()
