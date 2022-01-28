#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../../..)

while getopts ":h" option; do
    case $option in
    h | *) # display help
        echo
        echo "Syntax: bash run.sh GPU_ID MODEL_SIZE"
        echo "MODEL_SIZE choices: [base | large]"
        echo
        exit
        ;;
    esac
done

GPU=$1
MODEL_SIZE=${2:-base}

PATH_2_DATA=${CURRENT_DIR}/data/csnet/python

ARCH=mbart_${MODEL_SIZE}
PRETRAINED_MODEL_NAME=plbart_${MODEL_SIZE}.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX

SAVE_DIR=${CURRENT_DIR}/csnet_plbart_${MODEL_SIZE}
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/source

export CUDA_VISIBLE_DEVICES=$GPU

function fine_tune() {

    OUTPUT_FILE=${SAVE_DIR}/finetune.log

    # we have 374194 functions with docstrings (155186 standalone)
    # we use a batch size of 1024
    # if plbart_base: (per-gpu-batch-size=16; num-gpu=8, update-freq=8)
    # if plbart_large: (per-gpu-batch-size=8; num-gpu=8, update-freq=16)
    # lr 3e-04, max-update 20000

    fairseq-train $PATH_2_DATA/data-bin \
        --user-dir $USER_DIR \
        --langs $langs \
        --task translation_without_lang_token \
        --arch $ARCH \
        --layernorm-embedding \
        --truncate-source \
        --source-lang source \
        --target-lang target \
        --criterion cross_entropy \
        --bpe 'sentencepiece' \
        --sentencepiece-model $SPM_MODEL \
        --batch-size 16 \
        --update-freq 8 \
        --optimizer adam \
        --adam-eps 1e-06 \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay \
        --lr 3e-04 \
        --min-lr -1 \
        --warmup-updates 1000 \
        --max-update 20000 \
        --save-interval-updates 1000 \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --weight-decay 0.0 \
        --seed 1234 \
        --log-format json \
        --log-interval 10 \
        --restore-file $PRETRAIN \
        --reset-dataloader \
        --reset-optimizer \
        --reset-meters \
        --reset-lr-scheduler \
        --eval-bleu \
        --eval-bleu-detok space \
        --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-args '{"beam": 1}' \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --patience 10 \
        --no-epoch-checkpoints \
        --ddp-backend no_c10d \
        --save-dir $SAVE_DIR \
        --fp16 \
        2>&1 | tee "$OUTPUT_FILE"

}

fine_tune
