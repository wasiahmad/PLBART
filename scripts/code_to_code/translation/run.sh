#!/usr/bin/env bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../../..`;

PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX

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

PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-code/translation/data-bin

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${CURRENT_DIR}/${SOURCE}_${TARGET}
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/source


function fine_tune () {

OUTPUT_FILE=${SAVE_DIR}/finetune.log

# we have 10.3k train examples, use a batch size of 16 gives us 644 steps
# we run for a maximum of 50 epochs
# setting the batch size to 8 with update-freq to 2
# performing validation at every 500 steps, saving the last 10 checkpoints

fairseq-train $PATH_2_DATA \
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

fairseq-generate $PATH_2_DATA \
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
cat $FILE_PREF | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.ref
sacrebleu -tok 'none' -s 'none' $FILE_PREF.ref < $FILE_PREF.hyp 2>&1 | tee ${RESULT_FILE}

echo "CodeXGlue Evaluation: \t" >> ${RESULT_FILE}
python ${HOME_DIR}/evaluation/evaluator.py \
    --ref ${FILE_PREF}.ref \
    --pre ${FILE_PREF}.hyp \
    2>&1 | tee -a ${RESULT_FILE};

echo "CodeBLEU Evaluation" >> ${RESULT_FILE}
cd ${HOME_DIR}/evaluation/CodeBLEU;
python calc_code_bleu.py \
    --refs $FILE_PREF.ref \
    --hyp $FILE_PREF.hyp \
    --lang ${LANG_MAP[$TARGET]} \
    2>&1 | tee -a ${RESULT_FILE};
cd $CURRENT_DIR;

}

fine_tune
generate
