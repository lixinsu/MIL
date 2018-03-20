#!/usr/bin/env python
# coding: utf-8
import torch
from collections import Counter


def vectorize(ex, model):
    args = model.opt
    word_dict = model.word_dict
    #feature_dict = model.feature_dict

    # Index words , list of Tensor
    document = [torch.LongTensor([word_dict[w] for w in p]) for p in ex['passage']]

    label = ex['label']
    assert len(document) == len(label)

    while len(document) < args['sent_num']:
        document.append(torch.LongTensor([0]))
        label.append(0)
    # Tensor
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    # Tensor
    label =  torch.LongTensor(label)
    return document, question, label, ex['id']


def batchify(batch):
    docs = [ex[0] for ex in batch]
    questions = [ex[1] for ex in batch]
    ids = [ex[-1] for ex in batch]
    labels = [ex[2].unsqueeze(0) for ex in batch]

    # batch document and features
    max_length = max([len(y) for x in docs for y  in x])
    x1 = torch.LongTensor(len(docs), len(docs[0]), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), len(docs[0]), max_length).fill_(1)
    for i, ps in enumerate(docs):
        for j, p in enumerate(ps):
            x1[i, j, :p.size(0)].copy_(p)
            x1_mask[i, j, :p.size(0)].fill_(0)
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].fill_(0)
        x2_mask[i, :q.size(0)].fill_(0)

    labels = torch.cat(labels, dim=0)

    return x1, x1_mask, x2, x2_mask, labels, ids

