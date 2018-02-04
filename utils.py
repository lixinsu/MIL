#!/usr/bin/env python
# coding: utf-8

from sklearn import metrics
import torch

def get_n_params(model):
    """get number of model parameters"""
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def auc_para(pred_scores, gts):
    """auc paragraph"""
    pred_scores = [y for x in pred_scores for y in x]
    gts = [y for x in gts for y in x]
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    return metrics.auc(fpr, tpr)