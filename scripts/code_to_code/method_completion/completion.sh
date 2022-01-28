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
DATA_CHOICE=${3:-github}

if [[ $DATA_CHOICE == 'github' ]]; then
    SHARDS_DIR=${CURRENT_DIR}/data/github/python/shards
    PATH_2_DATA=""
    for ((idx = 0; idx <= 7; idx++)); do
        PATH_2_DATA+="${SHARDS_DIR}/shard${idx}"
        if [[ $idx -lt 7 ]]; then
            PATH_2_DATA+=":"
        fi
    done
else
    PATH_2_DATA=${CURRENT_DIR}/data/csnet/python/data-bin
fi

ARCH=mbart_${MODEL_SIZE}
PRETRAINED_MODEL_NAME=plbart_${MODEL_SIZE}.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
langs=java,python,en_XX

SAVE_DIR=${CURRENT_DIR}/plbart_${MODEL_SIZE}_${DATA_CHOICE}
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/source

if [[ -f $SAVE_DIR/checkpoint_last.pt ]]; then
    RESTORE_PARAMS="--restore-file $SAVE_DIR/checkpoint_last.pt"
else
    RESTORE_PARAMS="--restore-file $PRETRAIN "
    RESTORE_PARAMS+="--reset-optimizer "
    RESTORE_PARAMS+="--reset-meters "
    RESTORE_PARAMS+="--reset-dataloader "
    RESTORE_PARAMS+="--reset-lr-scheduler"
fi

# be default we use LOSS based validation
VALIDATION_ARGS=""
# OR, we may use BLEU based validation
#VALIDATION_ARGS="--eval-bleu "
#VALIDATION_ARGS+="--best-checkpoint-metric bleu "
#VALIDATION_ARGS+="--eval-bleu-print-samples "
#VALIDATION_ARGS+="--maximize-best-checkpoint-metric"

export CUDA_VISIBLE_DEVICES=$GPU

function fine_tune() {

    OUTPUT_FILE=${SAVE_DIR}/finetune.log

    # We have 60M functions in Github-python data
    # We want to use a batch size of 2048
    # So, adjust the MAX_TOKENS and UPDATE_FREQ accordingly
    # Tips: average length of code sequence is 320 tokens
    # So, in 11gb GPU, we can use MAX_TOKENS=2048

    # assuming 8 11gb GPUs
    if [ $MODEL_SIZE == "base" ]; then
        MAX_TOKENS=2048
        UPDATE_FREQ=40
    elif [ $MODEL_SIZE == "large" ]; then
        MAX_TOKENS=2048 # this may cause OOM
        UPDATE_FREQ=40
    fi

    fairseq-train $PATH_2_DATA \
        --user-dir $USER_DIR \
        --langs $langs \
        --dataset-impl 'mmap' \
        --task code_completion \
        --split-logic "random" \
        --arch $ARCH \
        --layernorm-embedding \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens $MAX_TOKENS \
        --max-sentences 32 \
        --update-freq $UPDATE_FREQ \
        --optimizer adam \
        --adam-eps 1e-06 \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay \
        --lr 3e-04 \
        --min-lr -1 \
        --warmup-updates 1000 \
        --max-update 100000 \
        --save-interval-updates 1000 \
        --skip-invalid-size-inputs-valid-test \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --weight-decay 0.0 \
        --seed 1234 \
        --log-format json \
        --log-interval 10 \
        --no-epoch-checkpoints \
        --ddp-backend no_c10d \
        --save-dir $SAVE_DIR \
        --fp16 \
        --eval-bleu-detok space \
        --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-args '{"beam": 1}' \
        $RESTORE_PARAMS \
        $VALIDATION_ARGS \
        2>&1 | tee "$OUTPUT_FILE"

}

fine_tune
