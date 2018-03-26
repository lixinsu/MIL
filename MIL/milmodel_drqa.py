#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from rnn_reader import RnnDocReader


class MIL_AnswerTrigger(nn.Module):
    def __init__(self, args):
        super(MIL_AnswerTrigger, self).__init__()
        args.word_embeddings = word_embeddings
        self.fill_args(args)
        self.args = args
        self.rnnreader = RnnDocReader(args)
        self.global_rnn = nn.GRU(args.hidden_size * args.doc_layers * 2 , args.hidden_size  * 2, num_layers=args.num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.linear1 = nn.Linear(args.hidden_size * 4 * 5, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 1)
        self.dropout_merge = nn.Dropout(0.5)
        self.dropout_fc = nn.Dropout(0.5)

    def fill_args(self, args):
        args.vocab_size = len(args.word_embeddings)
        args.embedding_dim = args.emb_dim
        args.use_qemb = True
        args.num_features = 0
        args.doc_layers = 2
        args.question_layers = 2
        args.dropout_emb = 0.4
        args.dropout_rnn = 0.4
        args.dropout_rnn_output = 0.4
        args.concat_rnn_layers = True
        args.rnn_type = 'gru'
        args.rnn_padding = False
        args.question_merge = 'self_attn'

    def forward(self, q, s, q_w_mask, s_w_mask):
        """
            q:
            s:
        """
        s = s.transpose(0, 1)
        s_w_mask = s_w_mask.transpose(0, 1)
        s_hiddens = []
        merge_hiddens = []
        for i in range(len(s)):  #for each passage
            doc_hidden, merge_hidden, q_hidden  = self.rnnreader(s[i], None, s_w_mask[i], q, q_w_mask)
            #print(q_hidden)
            #print('doc q merge')
            #print(torch.sum(merge_hidden.abs()))
            #print('doc_hidden')
            #print(torch.sum(doc_hidden.abs()))
            #print('q_hidden')
            #print(torch.sum(q_hidden.abs()))
            s_hiddens.append(merge_hidden)
            #merge_hiddens.append(merge_hidden)

        global_rnn_input = torch.cat([x.unsqueeze(1) for x in s_hiddens], dim=1)
        #p_global = torch.cat(list(self.global_rnn(global_rnn_input)[1]), dim=1)
        p_global = torch.max(global_rnn_input, dim=1)[0]
        scores = []
        for p_hidden in s_hiddens:
            merge = torch.cat([p_hidden, p_hidden - p_global, q_hidden, p_hidden*q_hidden, p_global*q_hidden], dim=1)
            out = self.dropout_merge(merge)
            out = self.linear1(out)
            out = self.dropout_fc(out)
            out = F.relu(out)
            score = self.linear2(out)
            scores.append(score)

        scores = torch.cat(scores, dim=1)
        if self.args.loss == 'bce':
            scores = F.sigmoid(scores)
        elif self.args.loss == 'merge':
            scores = F.softmax(scores)
        return scores


