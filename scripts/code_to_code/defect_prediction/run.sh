#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../../..)

while getopts ":h" option; do
    case $option in
    h | *) # display help
        echo
        echo "Syntax: bash run.sh GPU_ID"
        echo
        exit
        ;;
    esac
done

GPU=$1
MODEL_SIZE=${2:-base}
PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-code/defect_prediction/processed

ARCH=mbart_${MODEL_SIZE}
PRETRAINED_MODEL_NAME=plbart_${MODEL_SIZE}.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX

SAVE_DIR=${CURRENT_DIR}/devign
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/source

export CUDA_VISIBLE_DEVICES=$GPU

function fine_tune() {

    OUTPUT_FILE=${SAVE_DIR}/finetune.log

    MAX_UPDATES=15000  # ~11 epochs through 21,854 examples with bsz 16
    WARMUP_UPDATES=500 # 6 percent of the number of updates
    LR=5e-5            # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=4 # Batch size 16.
    UPDATE_FREQ=4

    fairseq-train $PATH_2_DATA/data-bin \
        --user-dir $USER_DIR \
        --restore-file $PRETRAIN \
        --max-positions 512 \
        --langs $langs \
        --task plbart_sentence_prediction \
        --add-prev-output-tokens \
        --shorten-method "truncate" \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer \
        --reset-dataloader \
        --reset-meters \
        --required-batch-size-multiple 1 \
        --init-token 0 \
        --arch $ARCH \
        --criterion sentence_prediction \
        --num-classes $NUM_CLASSES \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --weight-decay 0.0 \
        --optimizer adam \
        --adam-betas "(0.9, 0.98)" \
        --adam-eps 1e-08 \
        --clip-norm 1.0 \
        --lr-scheduler polynomial_decay \
        --lr $LR \
        --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --batch-size $MAX_SENTENCES \
        --max-epoch 10 \
        --max-update $MAX_UPDATES \
        --seed 1234 \
        --log-format json \
        --log-interval 10 \
        --best-checkpoint-metric accuracy \
        --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints \
        --find-unused-parameters \
        --ddp-backend no_c10d \
        --save-dir $SAVE_DIR \
        2>&1 | tee ${OUTPUT_FILE}

}

function generate() {

    RESULT_FILE=${SAVE_DIR}/result.txt
    python evaluator.py \
        --user_dir $USER_DIR \
        --model_dir $SAVE_DIR \
        --model_name checkpoint_best.pt \
        --data_bin_path ${PATH_2_DATA}/data-bin \
        --input_file ${PATH_2_DATA}/test.input0 \
        --label_file ${PATH_2_DATA}/test.label \
        --batch_size 128 \
        --max_example -1 \
        --output $RESULT_FILE

}

fine_tune
generate
