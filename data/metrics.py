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


def auc_para(pred_scores, gts):
    """auc paragraph"""
    pred_scores = [y for x in pred_scores for y in x]
    gts = [y for x in gts for y in x]
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    return metrics.auc(fpr, tpr), metrics.auc(recall, precision)



def auc_question(pred_scores, gts):
    """question level auc"""
    pred_scores =[max(x) for x in pred_scores]
    gts = [max(x) for x in gts]
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    return metrics.auc(fpr, tpr), metrics.auc(recall, precision)


def F1_para(pred_scores, gts, thresh=0.5):
    """auc paragraph"""
    pred_labels = [(1 if y > thresh else 0) for x in pred_scores for y in x]
    gts = [y for x in gts for y in x]
    return metrics.f1_score(gts, pred_labels)

def F1_question(pred_scores, gts, thresh=0.5):
    """question level auc"""
    pred_labels =[(1 if max(x) > thresh else 0) for x in pred_scores]
    gts = [max(x) for x in gts]
    return metrics.f1_score(gts, pred_labels)
