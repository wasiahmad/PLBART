#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

PRETRAINED_MODEL_NAME=checkpoint_356_100000.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX,javascript,php,ruby,go

declare -A LANG_MAP
LANG_MAP['java']='java'
LANG_MAP['cs']='c_sharp'

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID SRC_LANG TGT_LANG"
            echo "SRC_LANG/TGT_LANG  Language choices: [$(IFS=\| ; echo "${!LANG_MAP[@]}")]"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1

SOURCE=$2
TARGET=$3

PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-code/translation
CB_EVAL_SCRIPT=${HOME_DIR}/evaluation/CodeBLEU/calc_code_bleu.py

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${CURRENT_DIR}/code_to_code/translation/${SOURCE}_${TARGET}
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/source


function fine_tune () {

OUTPUT_FILE=${SAVE_DIR}/finetune.log

# we have 10.3k train examples, use a batch size of 16 gives us 644 steps
# we run for a maximum of 50 epochs
# setting the batch size to 8 with update-freq to 2
# performing validation at every 500 steps, saving the last 10 checkpoints

fairseq-train $PATH_2_DATA/data-bin \
    --user-dir $USER_DIR \
    --langs $langs \
    --task translation_without_lang_token \
    --arch mbart_base \
    --layernorm-embedding \
    --truncate-source \
    --source-lang $SOURCE \
    --target-lang $TARGET \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --batch-size 4 \
    --update-freq 4 \
    --max-epoch 30 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --lr 5e-05 \
    --min-lr -1 \
    --warmup-updates 1000 \
    --max-update 50000 \
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
GOUND_TRUTH_PATH=$PATH_2_DATA/test.java-cs.txt.${TARGET}

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
export PYTHONPATH=${HOME_DIR};
python $CB_EVAL_SCRIPT \
    --refs $GOUND_TRUTH_PATH \
    --hyp $FILE_PREF.hyp \
    --lang ${LANG_MAP[$TARGET]} \
    2>&1 | tee -a ${RESULT_FILE};

}

fine_tune
generate
