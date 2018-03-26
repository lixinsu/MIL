#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import fire

def merge_result(result_file, data_file, merge_file):
    outf = open(merge_file, 'w')
    examples = [json.loads(line) for line in open(data_file)]
    results = [json.loads(line) for line in open(result_file)]
    assert len(examples) == len(results)
    for result, example in zip(results, examples):
        assert example['query_id'] == result['query_id']
        outf.write('newcase:%s\n' % example['query_id'])
        outf.write('QUESTION: %s\n' % example['query'])
        outf.write('ANSWER: %s\n' % example['answer'])
        assert len(result['ground_truth']) >= len(example['passages'])
        scores = result['predictions']
        gts = result['ground_truth']
        for i, p in enumerate(example['passages']):
            outf.write('NO.%s \033[41;37m%.4f\033[0m %d %s\n' % (i,scores[i], gts[i], p['passage_text']))


if __name__ == '__main__':
    fire.Fire()

