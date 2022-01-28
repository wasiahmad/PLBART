#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../)

DATA_HOME=${HOME_DIR}/data
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX

DATA_DIR=""
for ((idx = 0; idx <= 7; idx++)); do
    DATA_DIR+="${DATA_HOME}/shards/shard${idx}"
    if [[ $idx -lt 7 ]]; then
        DATA_DIR+=":"
    fi
done

SAVE_DIR=${HOME_DIR}/pretrain
mkdir -p $SAVE_DIR
TENSORBOARD_LOGDIR=${SAVE_DIR}/tensorboard_logs

MAX_UPDATE=100000
WARMUP_UPDATES=2000
MAX_SENTENCES=32
MAX_TOKENS=2048
TOKENS_PER_SAMPLE=512
UPDATE_FREQ=60

export CUDA_VISIBLE_DEVICES=$1
fairseq-train $DATA_DIR \
    --add-lang-token \
    --langs $langs \
    --dataset-impl 'mmap' \
    --bpe 'sentencepiece' \
    --sentencepiece-model $SPM_MODEL \
    --arch mbart_base \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --max-tokens $MAX_TOKENS \
    --max-sentences $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ \
    --layernorm-embedding \
    --multilang-sampling-alpha 0.3 \
    --train-subset train \
    --valid-subset valid \
    --required-batch-size-multiple 8 \
    --insert 0 \
    --permute-sentences 0 \
    --poisson-lambda 3.5 \
    --mask 0.3 \
    --mask-length 'span-poisson' \
    --replace-length 1 \
    --rotate 0 \
    --mask-random 0.1 \
    --task multilingual_denoising \
    --criterion cross_entropy \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --relu-dropout 0.0 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --clip-norm 0.1 \
    --lr 3e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $MAX_UPDATE \
    --max-update $MAX_UPDATE \
    --fp16 \
    --ddp-backend=no_c10d \
    --no-epoch-checkpoints \
    --save-interval-updates 1000 \
    --keep-interval-updates 10 \
    --save-dir $SAVE_DIR \
    --skip-invalid-size-inputs-valid-test \
    --log-format json \
    --log-interval 10 \
    --num-workers 4 \
    --seed 1234 \
    --restore-file $SAVE_DIR/checkpoint_last.pt \
    --tensorboard-logdir $TENSORBOARD_LOGDIR \
    2>&1 | tee $SAVE_DIR/output.log
