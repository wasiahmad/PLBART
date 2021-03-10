#!/usr/bin/env bash

BASE_DIR=/local/username/codebart
HOME_DIR=/home/username/workspace/projects/CodeBART
langs=java,python,en_XX
declare -A LANG_MAP

LANG_MAP['concode']='java'
LANG_MAP['conala']='python'
LANG_MAP['conala-mined']='python'

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID DATASET_NAME"
            echo
            echo "DATASET_NAME choices: [concode, conala, conala-mined]"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2

if [[ $DATASET == 'concode' ]]; then
    BATCH_SIZE=8; UPDATE_FREQ=3; WARMUP=1000
    MAX_EPOCH=30; PATIENCE=10
elif [[ $DATASET == 'conala-mined' ]]; then
    BATCH_SIZE=8; UPDATE_FREQ=8; WARMUP=5000
    MAX_EPOCH=5; PATIENCE=3
else
    BATCH_SIZE=8; UPDATE_FREQ=2; WARMUP=1000
    MAX_EPOCH=50; PATIENCE=50
fi

SOURCE=en_XX
TARGET=${LANG_MAP[$DATASET]}

PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt
PRETRAIN=${BASE_DIR}/checkpoints/${PRETRAINED_MODEL_NAME}

PATH_2_DATA=${HOME_DIR}/data/codeXglue/text-to-code/${DATASET}/data-bin
SPM_MODEL=${BASE_DIR}/sentencepiece.bpe.model

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${BASE_DIR}/text-to-code/${DATASET}/${SOURCE}_${TARGET}-epoch_${MAX_EPOCH}
mkdir -p ${SAVE_DIR}


function fine_tune () {
	OUTPUT_FILE=${SAVE_DIR}/finetune.log

	fairseq-train $PATH_2_DATA \
        --restore-file $PRETRAIN \
        --bpe 'sentencepiece' \
        --sentencepiece-model $SPM_MODEL \
        --langs $langs --arch mbart_base --layernorm-embedding \
        --task translation_from_pretrained_bart \
        --source-lang $SOURCE --target-lang $TARGET \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
        --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --max-epoch $MAX_EPOCH \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay --lr 5e-05 --min-lr -1 \
        --warmup-updates $WARMUP --max-update 200000 \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --seed 1234 --log-format json --log-interval 100 \
        --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
        --eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints --patience $PATIENCE \
        --ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE}
}


function generate () {
	model=${SAVE_DIR}/checkpoint_best.pt
	FILE_PREF=${SAVE_DIR}/output
	RESULT_FILE=${SAVE_DIR}/result.txt

	if [[ $DATASET == 'conala' ]]; then
	    BEAM_SIZE=10
	    LEN_PEN=1
	else
	    BEAM_SIZE=10
	    LEN_PEN=1
	fi

	fairseq-generate $PATH_2_DATA \
  		--path $model \
  		--task translation_from_pretrained_bart \
  		--gen-subset test \
  		-t $TARGET -s $SOURCE \
  		--sacrebleu --remove-bpe 'sentencepiece' \
  		--batch-size 4 --langs $langs \
  		--beam $BEAM_SIZE --lenpen $LEN_PEN > $FILE_PREF

	cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp
	cat $FILE_PREF | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.ref
	sacrebleu -tok 'none' -s 'none' $FILE_PREF.ref < $FILE_PREF.hyp 2>&1 | tee ${RESULT_FILE}
	printf "CodeXGlue Evaluation: \t" >> ${RESULT_FILE}
	python evaluator.py --expected ${FILE_PREF}.ref --predicted ${FILE_PREF}.hyp
	python evaluator.py --expected ${FILE_PREF}.ref --predicted ${FILE_PREF}.hyp >> ${RESULT_FILE}
	echo "CodeBLEU Evaluation" > ${RESULT_FILE}
	cd CodeBLEU;
	python calc_code_bleu.py --refs $FILE_PREF.ref --hyp $FILE_PREF.hyp --lang $TARGET >> ${RESULT_FILE}
	python calc_code_bleu.py --refs $FILE_PREF.ref --hyp $FILE_PREF.hyp --lang $TARGET
	cd ..;
}

fine_tune
generate
