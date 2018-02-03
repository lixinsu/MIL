#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def ss(q):
    print(q.shape)


class MIL_AnswerTrigger(nn.Module):
    def __init__(self, args, word_embeddings):
        super(MIL_AnswerTrigger, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = False
        self.embedding.cuda()
        self.q_rnn = nn.GRU(args.emb_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.p_rnn = nn.GRU(args.emb_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.linear1 = nn.Linear(args.hidden_size * 2 * 2, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        self.dropout_emb = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, ex):
        # ex = [q, s, label, s_label, q_w_mask, s_w_mask, s_mask]
        q = Variable(torch.LongTensor(ex[0]).cuda())
        s = Variable(torch.LongTensor(ex[1]).cuda()).transpose(0, 1)  # change sn and bn
        s_emb = []
        for i in range(len(s)):
            s_emb.append(self.dropout_emb(self.embedding(s[i])))
        q_emb = self.dropout_emb(self.embedding(q))
        q_hidden = torch.cat(list(self.q_rnn(q_emb)[1]), dim=1)
        p_hiddens = [torch.cat(list(self.p_rnn(x)[1]), dim=1) for x in s_emb]
        scores = []
        for p_hidden in p_hiddens:
            merge = torch.cat([p_hidden, q_hidden], dim=1)
            out = self.dropout(merge)
            out = self.linear1(out)
            out = self.dropout_fc(out)
            out = F.relu(out)
            score = self.linear2(out)
            scores.append(F.sigmoid(score))
        return scores
