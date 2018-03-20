#!/usr/bin/env python
# coding: utf-8

import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from .milmodel import MIL_AnswerTrigger
from data.utils import AverageMeter


logger = logging.getLogger(__name__)


class MIL_Model(object):
    def __init__(self, opt, word_dict, state_dict=None):
        ori = opt
        self.word_dict = word_dict
        opt = vars(opt)
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        ori.vocab_size = len(word_dict)
        ori.emb_dim = 300
        self.network = MIL_AnswerTrigger(ori)
        def get_n_params(model):
            pp=0
            for p in list(model.parameters()):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                pp += nn
            return pp
        logger.info('number of parameters %s' % get_n_params(self.network))
        model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info('number of trainable params %s' % params)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network']):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            #self.optimizer = optim.Adamax(parameters,
            #                              weight_decay=opt['weight_decay'])
            self.optimizer = torch.optim.Adam(parameters)
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        logger.info('===========model structure============')
        logger.info(repr(self.network))
        logger.info('total model patameters %s' % get_n_params(self.network))
        model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info('trainable params : %s ' % params)

    def update(self, ex):
        """train a batch"""
        self.network.train()
        inputs = [Variable(ex[0].cuda()), Variable(ex[1].cuda()), Variable(ex[2].cuda()), Variable(ex[3].cuda())]
        target = Variable(ex[4].cuda())
        score = self.network(*inputs)

        if self.opt['loss'] == 'merge':
            loss = self.merge_loss(score, target)
        elif self.opt['loss'] == 'bce':
            loss = self.loss(score, target)
        self.optimizer.zero_grad()
        self.train_loss.update(loss.data[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])
        self.optimizer.step()
        self.updates += 1
        return loss.data[0]

    def predict(self, ex):
        """predict a batch"""
        self.network.eval()
        inputs = [Variable(ex[0].cuda()), Variable(ex[1].cuda()), Variable(ex[2].cuda()), Variable(ex[3].cuda())]
        target = Variable(ex[4].cuda())
        score = self.network(*inputs)
        if self.opt['loss'] == 'merge':
            loss = self.merge_loss(score, target)
        elif self.opt['loss'] == 'bce':
            loss = self.loss(score, target)
        else:
            raise 'unsupport loss'
        score = score.cpu().data.numpy().tolist()
        target = target.cpu().data.numpy().tolist()
        return score, target, loss.data[0]

    def load_embeddings(self, words, embedding_file):
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                (len(words), embedding_file))
        embedding = self.network.embedding.weight.data
        self.network.requires_grad = False
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                                'WARN: Duplicate embedding found for %s' % w
                                )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)
        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)
        logger.info('Loaded %d embeddings (%.2f%%)' %
                (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def loss(self, score, target):
        final_loss = self.weighted_binary_cross_entropy(score.view(-1), target.view(-1), weights=[1,1])
        return final_loss

    def weighted_binary_cross_entropy(self, output, target, weights=None):
        target = target.float()
        if weights is not None:
            assert len(weights) == 2

            loss = weights[1] * (target * torch.log(output)) + \
                   weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))

    def merge_loss(self, score, target):
        target = target.float()
        pos_bag_mask= torch.max(target, dim=1)[0]
        neg_bag_mask = 1 - pos_bag_mask
        neg_margin = self.opt['neg_margin'] - (0.5 - neg_bag_mask * torch.max(score, dim=1)[0]).unsqueeze(1)
        cost_neg = torch.max(torch.cat([neg_margin, Variable(torch.zeros(target.size(0)).cuda()).unsqueeze(1)], dim=1), dim=1)[0]
        avg_cost_neg = torch.sum(cost_neg) / (torch.sum(neg_bag_mask) + 1e-12)

        pos_pred = score * target
        neg_pred = score * (1 - target)
        max_pos_pred = torch.max(pos_pred, dim=1)[0]
        max_neg_pred = torch.max(neg_pred, dim=1)[0]
        PN_margin = self.opt['pos_neg_margin'] - (max_pos_pred - max_neg_pred).unsqueeze(1)
        cost_pn = pos_bag_mask * torch.max(torch.cat([PN_margin, Variable(torch.zeros(target.size(0)).cuda()).unsqueeze(1)], dim=1), dim=1)[0]
        avg_cost_pn = torch.sum(cost_pn) / (torch.sum(pos_bag_mask) + 1e-12)

        pos_margin = self.opt['pos_margin'] - (max_pos_pred - 0.5).unsqueeze(1)
        cost_pos = pos_bag_mask * torch.max(torch.cat([pos_margin, Variable(torch.zeros(target.size(0)).cuda()).unsqueeze(1)], dim=1), dim=1)[0]
        avg_cost_pos = torch.sum(cost_pos) / (torch.sum(pos_bag_mask) + 1e-12)
        #print(avg_cost_neg.data[0], avg_cost_pn.data[0], avg_cost_pos.data[0])
        final_loss = avg_cost_neg + avg_cost_pn + avg_cost_pos
        #print(final_loss.data[0])
        return final_loss


    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename + str(epoch))
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')
