#!/bin/bash

set -ex

TASK=searchqa
TOKENIZER=spacy
THREAD=2

#python3 scripts/preprocess.py ${TASK}/ ${TASK}/ sample_train --tokenizer ${TOKENIZER}  --workers ${THREAD}
#python3 scripts/preprocess.py ${TASK}/ ${TASK}/ sample_val --tokenizer ${TOKENIZER}  --workers ${THREAD}
#python3 scripts/preprocess.py ${TASK}/ ${TASK}/ sample_test --tokenizer ${TOKENIZER}  --workers ${THREAD}

python3 scripts/train.py  --data-dir ${TASK} \
                          --task ${TASK} \
                          --loss bce \
i                         --log-file ${TASK}_output.log
