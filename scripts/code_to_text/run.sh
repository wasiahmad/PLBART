#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../..)

while getopts ":h" option; do
    case $option in
    h | *) # display help
        echo
        echo "Syntax: bash run.sh GPU_ID SRC_LANG"
        echo
        echo "SRC_LANG  Language choices: [java, python, go, javascript, php, ruby]"
        echo
        exit
        ;;
    esac
done

GPU=$1
SOURCE=$2
MODEL_SIZE=${3:-base}

TARGET=en_XX
PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-text/${SOURCE}

ARCH=mbart_${MODEL_SIZE}
PRETRAINED_MODEL_NAME=plbart_${MODEL_SIZE}.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX

SAVE_DIR=${CURRENT_DIR}/${MODEL_SIZE}_${SOURCE}_${TARGET}
mkdir -p ${SAVE_DIR}

if [[ "$SOURCE" =~ ^(ruby|javascript|go|php)$ ]]; then
    USER_DIR="--user-dir ${HOME_DIR}/source"
    TASK=translation_without_lang_token
else
    USER_DIR=""
    TASK=translation_from_pretrained_bart
fi

export CUDA_VISIBLE_DEVICES=$GPU

function fine_tune() {

    OUTPUT_FILE=${SAVE_DIR}/finetune.log
    fairseq-train $PATH_2_DATA/data-bin $USER_DIR \
        --restore-file $PRETRAIN \
        --bpe 'sentencepiece' \
        --sentencepiece-model $SPM_MODEL \
        --langs $langs \
        --arch $ARCH \
        --layernorm-embedding \
        --task $TASK \
        --source-lang $SOURCE \
        --target-lang $TARGET \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.2 \
        --batch-size 8 \
        --update-freq 4 \
        --max-epoch 15 \
        --optimizer adam \
        --adam-eps 1e-06 \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay \
        --lr 5e-05 \
        --min-lr -1 \
        --warmup-updates 1000 \
        --max-update 200000 \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --weight-decay 0.0 \
        --seed 1234 \
        --log-format json \
        --log-interval 100 \
        --reset-optimizer \
        --reset-meters \
        --reset-dataloader \
        --reset-lr-scheduler \
        --eval-bleu \
        --eval-bleu-detok space \
        --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-args '{"beam": 5}' \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints \
        --patience 5 \
        --ddp-backend no_c10d \
        --save-dir $SAVE_DIR \
        2>&1 | tee ${OUTPUT_FILE}

}

function generate() {

    model=${SAVE_DIR}/checkpoint_best.pt
    FILE_PREF=${SAVE_DIR}/output
    RESULT_FILE=${SAVE_DIR}/result.txt
    GOUND_TRUTH_PATH=$PATH_2_DATA/test.jsonl

    fairseq-generate $PATH_2_DATA/data-bin $USER_DIR \
        --path $model \
        --task $TASK \
        --gen-subset test \
        -t $TARGET -s $SOURCE \
        --sacrebleu \
        --remove-bpe 'sentencepiece' \
        --batch-size 8 \
        --langs $langs \
        --beam 10 >$FILE_PREF

    cat $FILE_PREF | grep -P "^H" | sort -V | cut -f 3- | sed 's/\[${TARGET}\]//g' >$FILE_PREF.hyp
    python ${HOME_DIR}/evaluation/nl_eval.py \
        --references $GOUND_TRUTH_PATH \
        --predictions $FILE_PREF.hyp \
        --json_refs 2>&1 | tee ${RESULT_FILE}

}

fine_tune
generate
