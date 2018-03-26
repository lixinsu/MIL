#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
import json
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from visdom import Visdom
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from MIL.model import MIL_Model
from data.metrics import AverageMeter, get_n_params, auc_para, auc_question, F1_para, F1_question
from data import utils, vector, data, visual_helper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='Batch size during validation/testing')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')

    # files
    parser.add_argument('--data-dir', default='quasart', help='data dir')
    parser.add_argument('--out-dir', default='quasart/pred', help='data dir')
    parser.add_argument('--log-file', default='quasart_test_output.log', type=str, help='test log file')
    parser.add_argument('--test-file', default='sample_test-processed-spacy.txt', help='train data file')
    parser.add_argument('--dev-file', default='sample_val-processed-spacy.txt', help='train data file')
    parser.add_argument('--out-file', default='quasart_result.txt', help='train data file')
    parser.add_argument('--model-file', default='saved_models/quasart', type=str, help='save model directory')
    parser.add_argument('--sent-num', type=int, default=10, help='number of sentences')
    parser.add_argument('--data-workers', type=int, default=10, help='number of data prepare works')
    return parser.parse_args()

args = parse_args()

args.test_file = os.path.join(args.data_dir, args.test_file)
args.dev_file = os.path.join(args.data_dir, args.dev_file)

 # setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s  [%(filename)s,%(lineno)d-%(funcName)s]', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def evaluate(data_loader, model, result_file=None):
    pred_scores = []
    gts = []
    ids = []
    for idx, ex in tqdm(enumerate(data_loader)):
        ids.extend(ex[-1])
        score, target, loss = model.predict(ex)
        pred_scores.extend([[round(x,3) for x in y] for y in score])
        gts.extend(target)
    auc_para_roc, auc_para_pr = auc_para(pred_scores, gts)
    auc_q_roc, auc_q_pr = auc_question(pred_scores, gts)
    logger.info('auc_pr_Q %s' % auc_q_pr)
    logger.info('auc_pr_P %s' % auc_para_pr)
    if result_file is not None:
        outf = open(result_file, 'w')
        logger.info('save prediction file %s' % result_file)
        for score, target, qid in zip(pred_scores, gts, ids):
            outf.write(json.dumps({'query_id': qid, 'predictions': score, 'ground_truth': target}) + '\n')
    return {'auc_q_roc' : auc_q_roc, 'auc_q_pr': auc_q_pr, 'auc_para_roc': auc_para_roc, 'auc_para_pr': auc_para_pr}


def predict():

    logger.info('-' * 100)
    logger.info('loading model')
    params = torch.load(args.model_file)
    model = MIL_Model(params['config'],params['word_dict'],\
            state_dict=params['state_dict'])
    model.gpu()
    logger.info('Load data files')
    test_exs = utils.load_data(args, args.test_file)
    logger.info('Num train examples = %d' % len(test_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))
    logger.info('-' * 100)
    logger.info('Training model from scratch...')

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev.
    test_dataset = data.ReaderDataset(test_exs, model)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            sampler=test_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
    )
    dev_dataset = data.ReaderDataset(dev_exs, model)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
    )
    # --------------------------------------------------------------------------
    # PREDICT

    logger.info('predicting test dataset')
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logger.info('evaluating test set')
    evaluate(test_loader, model, os.path.join(args.out_dir, args.out_file))
    logger.info('evaluating dev set')
    evaluate(dev_loader, model)


if __name__ == "__main__":
    predict()
