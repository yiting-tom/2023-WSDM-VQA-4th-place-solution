#!/usr/bin/env bash

########################## Evaluate VG ##########################
data=../dataset/vg.tsv
user_dir=../ofa_module
bpe_dir=../utils/BPE
selected_cols=0,4,2,3
split="vg"

model="checkpoint.pt"

path=../wsdm_checkpoints/${model}
result_path=../dataset

python3 ../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=12 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=1 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"