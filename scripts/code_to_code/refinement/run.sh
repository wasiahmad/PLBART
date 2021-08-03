#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../../..`;

PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID DATA_SIZE"
            echo
            echo "DATA_SIZE: small, medium"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1
DATA_SIZE=$2

SOURCE=source
TARGET=target

PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-code/refinement/${DATA_SIZE}

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${CURRENT_DIR}/${DATA_SIZE}
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/source

if [[ $DATA_SIZE == 'small' ]]; then
    BATCH_SIZE=16; UPDATE_FREQ=1;
else
    BATCH_SIZE=8; UPDATE_FREQ=2;
fi


function fine_tune () {

OUTPUT_FILE=${SAVE_DIR}/finetune.log

# approx. 50k train examples, use a batch size of 16 gives us 3000 steps
# we run for a maximum of 30 epochs
# setting the batch size to 8 with update-freq to 2
# performing validation at every 2000 steps, saving the last 10 checkpoints

fairseq-train $PATH_2_DATA/data-bin \
    --user-dir $USER_DIR \
    --truncate-source \
    --langs $langs \
    --task translation_without_lang_token \
    --arch mbart_base \
    --layernorm-embedding \
    --source-lang $SOURCE \
    --target-lang $TARGET \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --batch-size $BATCH_SIZE \
    --update-freq $UPDATE_FREQ \
    --max-epoch 30 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --lr 5e-05 --min-lr -1 \
    --warmup-updates 500 \
    --max-update 100000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.0 \
    --seed 1234 \
    --log-format json \
    --log-interval 100 \
    --restore-file $PRETRAIN \
    --reset-dataloader \
    --reset-optimizer \
    --reset-meters \
    --reset-lr-scheduler \
    --eval-bleu \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 5}' \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --patience 10 \
    --ddp-backend no_c10d \
    --save-dir $SAVE_DIR \
    2>&1 | tee ${OUTPUT_FILE};

}


function generate () {

model=${SAVE_DIR}/checkpoint_best.pt
FILE_PREF=${SAVE_DIR}/output
RESULT_FILE=${SAVE_DIR}/result.txt
GOUND_TRUTH_PATH=$PATH_2_DATA/test.buggy-fixed.fixed

fairseq-generate $PATH_2_DATA/data-bin \
    --user-dir $USER_DIR \
    --path $model \
    --truncate-source \
    --task translation_without_lang_token \
    --gen-subset test \
    -t $TARGET -s $SOURCE \
    --sacrebleu \
    --remove-bpe 'sentencepiece' \
    --max-len-b 200 \
    --beam 5 \
    --batch-size 4 \
    --langs $langs > $FILE_PREF

cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp

echo "CodeXGlue Evaluation" > ${RESULT_FILE}
python ${HOME_DIR}/evaluation/bleu.py \
    --ref $GOUND_TRUTH_PATH \
    --pre $FILE_PREF.hyp \
    2>&1 | tee -a ${RESULT_FILE};

echo "CodeBLEU Evaluation" >> ${RESULT_FILE}
cd ${HOME_DIR}/evaluation/CodeBLEU;
PYTHONPATH=${HOME_DIR} python calc_code_bleu.py \
    --refs $GOUND_TRUTH_PATH \
    --hyp $FILE_PREF.hyp \
    --lang java \
    2>&1 | tee -a ${RESULT_FILE}
cd $CURRENT_DIR;

}

fine_tune
generate
