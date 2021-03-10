#!/usr/bin/env bash

BASE_DIR=/local/username/codebart
HOME_DIR=/home/username/workspace/projects/CodeBART
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

#checkpoint_3_25000.pt  checkpoint_6_50000.pt  checkpoint_8_75000.pt checkpoint_11_100000.pt
PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt
PRETRAIN=${BASE_DIR}/checkpoints/${PRETRAINED_MODEL_NAME}

PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-code/refinement/${DATA_SIZE}/data-bin

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${BASE_DIR}/code-to-code/refinement/${DATA_SIZE}-${PRETRAINED_MODEL_NAME}-epoch_30
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/src

if [[ $DATA_SIZE == 'small' ]];
then
    BATCH_SIZE=16
    UPDATE_FREQ=1
else
    BATCH_SIZE=8
    UPDATE_FREQ=2
fi


function fine_tune () {
	OUTPUT_FILE=${SAVE_DIR}/finetune.log

	# approx. 50k train examples, use a batch size of 16 gives us 3000 steps
	# we run for a maximum of 30 epochs
	# setting the batch size to 8 with update-freq to 2
	# performing validation at every 2000 steps, saving the last 10 checkpoints

	fairseq-train $PATH_2_DATA \
	    --user-dir $USER_DIR --truncate-source \
  		--langs $langs --task translation_in_same_language \
  		--arch mbart_base --layernorm-embedding \
  		--source-lang $SOURCE --target-lang $TARGET \
  		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  		--batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --max-epoch 30 \
  		--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  		--lr-scheduler polynomial_decay --lr 5e-05 --min-lr -1 \
  		--warmup-updates 500 --max-update 100000 \
  		--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  		--seed 1234 --log-format json --log-interval 100 \
  		--restore-file $PRETRAIN --reset-dataloader \
  		--reset-optimizer --reset-meters --reset-lr-scheduler \
  		--eval-bleu --eval-bleu-detok space --eval-tokenized-bleu \
  		--eval-bleu-remove-bpe sentencepiece --eval-bleu-args '{"beam": 5}' \
  		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  		--no-epoch-checkpoints --patience 10 \
  		--ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE}
}


function generate () {
	model=${SAVE_DIR}/checkpoint_best.pt
	FILE_PREF=${SAVE_DIR}/output
	RESULT_FILE=${SAVE_DIR}/result.txt

	fairseq-generate $PATH_2_DATA \
	    --user-dir $USER_DIR \
  		--path $model \
  		--truncate-source \
  		--task translation_in_same_language \
  		--gen-subset test \
  		-t $TARGET -s $SOURCE \
  		--sacrebleu --remove-bpe 'sentencepiece' \
		--max-len-b 200 --beam 5 \
  		--batch-size 4 --langs $langs > $FILE_PREF

	cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp
	cat $FILE_PREF | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.ref
	sacrebleu -tok 'none' -s 'none' $FILE_PREF.ref < $FILE_PREF.hyp 2>&1 | tee ${RESULT_FILE}
	printf "CodeXGlue Evaluation: \t" >> ${RESULT_FILE}
	python evaluator.py --ref ${FILE_PREF}.ref --pre ${FILE_PREF}.hyp
	python evaluator.py --ref ${FILE_PREF}.ref --pre ${FILE_PREF}.hyp >> ${RESULT_FILE}
	echo "CodeBLEU Evaluation" > ${RESULT_FILE}
	cd CodeBLEU;
	python calc_code_bleu.py --refs $FILE_PREF.ref --hyp $FILE_PREF.hyp --lang java >> ${RESULT_FILE}
	python calc_code_bleu.py --refs $FILE_PREF.ref --hyp $FILE_PREF.hyp --lang java
	cd ..;
}

fine_tune
generate
