#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import confusion_matrix, precision_score, f1_score


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


def evaluete_predictions(preds, dev_y):
    # instance level
    gts = [x[1] for x in dev_y]
    pred_labels = [(1 if x >= 0.5 else 0) for x in preds]
    print(confusion_matrix(gts, pred_labels))
    print(precision_score(gts,pred_labels))
    print(f1_score(gts, pred_labels))
    results = [(x[0], x[1], preds[i]) for i,x in enumerate(dev_y)]
    qid2info = {}
    for qid, label, score in results:
        if qid not in qid2info:
            qid2info[qid] = [[], []]
        qid2info[qid][0].append(label)
        qid2info[qid][1].append(score)
    bag_TP, bag_TN, bag_FP, bag_FN = 0, 0, 0, 0
    for k in qid2info:
        bag_pred = (1 if max(qid2info[k][1]) > 0.5 else 0)
        bag_gt = (1 if sum(qid2info[k][0]) >= 1 else 0)
        if bag_pred == bag_gt:
            if bag_pred == 1:
                bag_TP += 1
            else:
                bag_TN += 1
        else:
            if bag_pred == 1:
                bag_FP += 1
            else:
                bag_FN += 1
    precision = bag_TP * 1.0 / (bag_TP + bag_FP)
    recall = bag_TP * 1.0 / (bag_TP + bag_FN)
    f1 = 2 * precision * recall / (precision + recall)
    log.info('precision %.4f, recall %.4f f1:%.4f' % (precision, recall, f1))
    return f1
