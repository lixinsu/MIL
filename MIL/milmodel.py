#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def ss(q):
    print(q.shape)


class MIL_AnswerTrigger(nn.Module):
    def __init__(self, args):
        super(MIL_AnswerTrigger, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.emb_dim, padding_idx=0)
        #self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding.weight.requires_grad = False
        self.q_rnn = nn.GRU(args.emb_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.p_rnn = nn.GRU(args.emb_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.global_rnn = nn.GRU(args.hidden_size *2 , args.hidden_size, num_layers=args.num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.p_rnn = nn.GRU(args.emb_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.linear1 = nn.Linear(args.hidden_size * 2 * 5, args.hidden_size*2)
        self.linear2 = nn.Linear(args.hidden_size*2, 1)
        self.dropout_merge = nn.Dropout(0.5)
        self.dropout_emb = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.5)

    def fill_args(self, args):
        args.vocab

    def forward(self,s, s_w_mask, q, q_w_mask):
        s = s.transpose(0, 1)
        s_emb = []
        for i in range(len(s)):
            s_emb.append(self.dropout_emb(self.embedding(s[i])))
        q_emb = self.dropout_emb(self.embedding(q))
        q_hidden = torch.cat(list(self.q_rnn(q_emb)[1]), dim=1)
        p_hiddens = [torch.cat(list(self.p_rnn(x)[1]), dim=1) for x in s_emb]
        global_rnn_input = torch.cat([x.unsqueeze(1) for x in p_hiddens], dim=1)

        p_global = torch.mean(global_rnn_input, dim=1)
        #p_global = torch.cat(list(self.global_rnn(global_rnn_input)[1]), dim=1)
        scores = []
        for p_hidden in p_hiddens:
            merge = torch.cat([p_hidden, q_hidden, p_hidden * q_hidden, p_hidden - q_hidden, p_hidden - p_global], dim=1)
            out = self.dropout_merge(merge)
            out = self.linear1(out)
            out = self.dropout_fc(out)
            out = F.relu(out)
            score = self.linear2(out)
            #print (score)
            #print(score.size())
            scores.append(score)
        scores = torch.cat(scores, dim=1)
        #print(scores.size())
        if self.args.loss == 'bce':
            scores = F.sigmoid(scores)
        elif self.args.loss == 'merge':
            scores = F.softmax(scores, dim=1)
        # print(scores)
        return scores
