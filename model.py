#!/usr/bin/env python
# coding: utf-8

import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from milmodel import MIL_AnswerTrigger
from utils import AverageMeter, weighted_binary_cross_entropy, get_n_params


logger = logging.getLogger(__name__)


class MIL_Model(object):
    def __init__(self, opt, embedding=None, state_dict=None):
        ori = opt
        opt = vars(opt)
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        self.network = MIL_AnswerTrigger(ori, embedding).cuda()
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
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        logger.info('===========model structure============')
        logger.info(repr(self.network))
        logger.info('total model patameters %s' % get_n_params(self.network))
        model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info('trainable params : %s ' % params)

    def update(self, ex):
        self.network.train()
        inputs = [Variable(torch.LongTensor(ex[0]).cuda()), Variable(torch.LongTensor(ex[1]).cuda()), Variable(torch.LongTensor(ex[4]).cuda()), Variable(torch.LongTensor(ex[5]).cuda())]
        target = Variable(torch.Tensor(ex[3]).cuda())
        score = self.network(*inputs)
        loss = self.loss(score, target)
        self.optimizer.zero_grad()
        self.train_loss.update(loss.data[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])
        self.optimizer.step()
        self.updates += 1

    def predict(self, ex):
        self.network.eval()
        inputs = [Variable(torch.LongTensor(ex[0]).cuda()), Variable(torch.LongTensor(ex[1]).cuda()), Variable(torch.LongTensor(ex[4]).cuda()), Variable(torch.LongTensor(ex[5]).cuda())]
        # target = Variable(torch.Tensor(ex[3]).cuda(async=True))
        score = self.network(*inputs)
        score = score.cpu().data.numpy().tolist()
        return score

    @staticmethod
    def loss(score, target):
        return weighted_binary_cross_entropy(score.view(-1), target.view(-1), weights=[1, 4])

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
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')