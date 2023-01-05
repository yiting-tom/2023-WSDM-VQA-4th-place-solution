#!/usr/bin/env bash

# This script evaluates pretrained OFA-Large checkpoint on zero-shot open-domain VQA task.

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082
export CUDA_VISIBLE_DEVICES=0,1
export GPUS_PER_NODE=2

user_dir=../ofa_module
bpe_dir=../utils/BPE
########################## Evaluate VQA (zero-shot) ##########################
data=../dataset/vqa.tsv
path=../checkpoints/ofa_huge.pt
result_path=../dataset
selected_cols=0,5,2,3,4
subset=vqa

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../evaluate.py \
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
    --beam=1 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0