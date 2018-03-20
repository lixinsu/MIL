#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append(os.getcwd())
print(os.getcwd())
import json
import time
import argparse
import tokenizers
from multiprocessing import Pool
import copy

TOK = None
def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    #Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    global TOK
    tokens = TOK.tokenize(text)
    output = {
            'words': tokens.words(),
            'pos': tokens.pos(),
            'lemma': tokens.lemmas(),
            'ner': tokens.entities(),
    }
    return output


def tokenize_list(texts):
    global TOK
    outputs = []
    for text in texts:
        tokens = TOK.tokenize(text)
        output = {
                'words': tokens.words(),
                'pos': tokens.pos(),
                'lemma': tokens.lemmas(),
                'ner': tokens.entities(),
                }
        outputs.append(output)
    return outputs


def load_dataset(infile):
    output = {'qids':[], 'questions':[], 'contexts':[], 'answers':[], 'labels':[]}
    with open(infile) as f:
        for data in f:
            data = data.strip()
            if not data:
                continue
            data = json.loads(data)
            output['qids'].append(data['query_id'])
            output['questions'].append(data['query'])
            output['contexts'].append([x['passage_text'] for x in data['passages']])
            output['answers'].append(data['answer'])
            output['labels'].append([1 if data['answer'] in x['passage_text'] else 0 for x in data['passages']])
    return output


def process_dataset(data, tokenizer, num_workers=None):
    tokenizer_class = tokenizers.get_class(tokenizer)
    workers = Pool(num_workers, initializer=init,\
            initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()
    workers = Pool(num_workers, initializer=init,\
            initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}}))
    c_tokens = workers.map(tokenize_list, data['contexts'])
    workers.close()
    workers.join()
    for idx  in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        passage = [x['words'] for x in  c_tokens[idx]]
        q_pos = q_tokens[idx]['pos']
        q_ner = q_tokens[idx]['ner']
        p_pos = [x['pos'] for x in  c_tokens[idx]]
        p_ner = [x['ner'] for x in  c_tokens[idx]]
        label = data['labels'][idx]
        answer = data['answers'][idx]
        yield {
            'id': data['qids'][idx],
            'question':question,
            'passage':passage,
            'q_pos':q_pos,
            'q_ner':q_ner,
            'p_pos':p_pos,
            'p_ner':p_ner,
            'label':label,
            'answer':answer,
            }

# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='path to train dataset')
parser.add_argument('out_dir', type=str, help='path to output file dir')
parser.add_argument('split', type=str, help='filename for train dev dataset')
parser.add_argument('--tokenizer', type=str, default='spacy')
parser.add_argument('--workers', type=int, default=None)
args = parser.parse_args()


t0 = time.time()

infile = os.path.join(args.data_dir, args.split + '.json')
dataset = load_dataset(infile)
out_file = os.path.join(
        args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
        )
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')

print('Total time: %.4f (s)' % (time.time() - t0))
