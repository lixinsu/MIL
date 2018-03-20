#!/bin/bash

TASK=quasart


python3 scripts/preprocess.py ${TASK}/ ${TASK}/ sample_train --tokenizer spacy  --workers 10
python3 scripts/preprocess.py ${TASK}/ ${TASK}/ sample_val --tokenizer spacy  --workers 10
python3 scripts/preprocess.py ${TASK}/ ${TASK}/ sample_test --tokenizer spacy  --workers 10

python3 scripts/train.py 
