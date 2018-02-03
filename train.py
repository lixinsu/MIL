#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from data_load import load_data, BUCKETS, BatchData
from model import MIL_AnswerTrigger
from utils import AverageMeter, evaluete_predictions

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, help='train epochs')
    parser.add_argument('--batch_size', default=2, help='batch size')
    parser.add_argument('--hidden_size', default=128, help='rnn hidden size')
    parser.add_argument('--num_layers', default=1, help='rnn stacked layers')
    return parser.parse_args()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def train():
    args = parse_args()
    train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings = load_data('quasart/quasart.pkl')
    args.vocab_size = len(word2id)
    args.emb_dim = word_embeddings.shape[1]
    print(vars(args))
    assert (len(word2id) == len(word_embeddings))
    print('bucket statistics')
    for bid in range(len(BUCKETS)):
        print('bid:%s %s %s %s' % (bid, len(train_tuple_b[bid][0]), len(valid_tuple_b[bid][0]), len(test_tuple_b[bid][0])))
    batchdata = BatchData(train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings)
    mil_model = MIL_AnswerTrigger(args, word_embeddings).cuda()
    print(mil_model)
    print('total model patameters %s' % get_n_params(mil_model))
    num_step_in_epoch = 2000
    criterion = nn.BCELoss()
    parameters = [p for p in mil_model.parameters() if p.requires_grad]
    model_parameters = filter(lambda p: p.requires_grad, mil_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('trainable params : %s ' % params)
    optimizer = optim.Adamax(parameters)
    train_measure = AverageMeter()
    for epoch in range(args.epochs):
        losses = []
        mil_model.train()
        for step in range(num_step_in_epoch):
            # q, s, label, s_label, q_w_mask, s_w_mask, s_mask
            ex = batchdata.get_train_batch(0, args.batch_size)
        #for ex in batchdata.get_test_batch(0, args.batch_size, True):
            #print(ex)
            scores = mil_model(ex)
            s_label = Variable(torch.FloatTensor(ex[3])).cuda()
            scores = torch.cat(scores, dim=1)
            #print(scores[0])
            #print(s_label[0])
            loss = weighted_binary_cross_entropy(scores.view(-1), s_label.view(-1), weights=[1, 4])
            #print('loss>>>', loss.data[0])
            train_measure.update(loss.data[0])
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 100 == 0:
                print('train loss: %s' % train_measure.avg)
                train_measure.reset()

                test_losses = []
                mil_model.eval()
                preds = []
                labels = []
                for ex in batchdata.get_test_batch(0, 10, True):
                    scores = mil_model(ex)
                    s_label = Variable(torch.FloatTensor(ex[3])).cuda()
                    scores = torch.cat(scores, dim=1)
                    test_loss = criterion(scores.view(-1), s_label.view(-1))
                    preds.extend(scores.view(-1).data.cpu().numpy().tolist())
                    labels.extend(s_label.view(-1).data.cpu().numpy().tolist())
                    test_losses.append(test_loss.data[0])
                pred_labels = [(1 if x > 0.5 else 0) for x in preds]
                #print(preds)
                print('pred samples %s ' % len(pred_labels))
                print(f1_score(labels, pred_labels, average=None))
                print(confusion_matrix(labels, pred_labels))
                print(precision_score(labels, pred_labels, average=None))
                print("test loss %s " % (sum(test_losses) / len(test_losses)))

        print("epoch %s, train loss %s, " % (epoch, sum(losses) / len(losses)))


if __name__ == "__main__":
    train()
