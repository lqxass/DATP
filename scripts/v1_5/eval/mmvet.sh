#!/bin/bash

CKPT="llava-v1.5-7b"

TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${CKPT}/${PARAM}.jsonl \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${CKPT}/${PARAM}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${CKPT}/${PARAM}.json

