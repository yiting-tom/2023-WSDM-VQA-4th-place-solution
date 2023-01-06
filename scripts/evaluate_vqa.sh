#!/usr/bin/env bash

user_dir=../ofa_module
bpe_dir=../utils/BPE
########################## Evaluate VQA (zero-shot) ##########################
data=../dataset/vqa.tsv
path=../checkpoints/ofa_huge.pt
result_path=../dataset
selected_cols=0,5,2,3,4
subset=vqa

python3 ../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --patch-image-size=512 \
    --prompt-type='none' \
    --batch-size=12 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${subset} \
    --results-path=${result_path} \
    --fp16 \
    --zero-shot \
    --beam=30 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0
