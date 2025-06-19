#!/bin/bash

CKPT="llava-v1.5-7b"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa_loader \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}/${PARAM}.jsonl \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT}/${PARAM}

cd eval_tool

python calculation.py --results_dir answers/${CKPT}/${PARAM}
