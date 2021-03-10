#!/usr/bin/env bash

langs=java,python,en_XX
declare -A LANG_MAP

LANG_MAP['java']='java'
LANG_MAP['python']='python'
LANG_MAP['en']='en_XX'

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID SRC_LANG"
            echo
            echo "SRC_LANG  Language choices: [$(IFS=\| ; echo "${!LANG_MAP[@]}")]"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1
SRC_LANG=$2
TGT_LANG=en

FINETUNED_MODEL_NAME=$3

SOURCE=${LANG_MAP[$SRC_LANG]}
TARGET=${LANG_MAP[$TGT_LANG]}

PRETRAINED_CP_NAME=checkpoint_10_100000.pt


PRETRAIN=/local/username/codebart/checkpoints/${PRETRAINED_CP_NAME}
PATH_2_DATA=/home/username/workspace/projects/CodeBART/data/codeXglue/code-to-text/${SOURCE}/data-bin

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=/local/username/cbart/nmt-finetuned/${SOURCE}_${TARGET}-${PRETRAINED_CP_NAME}/${FINETUNED_MODEL_NAME}
mkdir -p ${SAVE_DIR}-output


function generate () {
	model=${SAVE_DIR}
	FILE_PREF=${SAVE_DIR}-output/output
	RESULT_FILE=${SAVE_DIR}-output/result.txt

	fairseq-generate $PATH_2_DATA \
 		--path $model \
  		--task translation_from_pretrained_bart \
  		--gen-subset test \
  		-t $TARGET -s $SOURCE \
  		--sacrebleu --remove-bpe 'sentencepiece' \
  		--batch-size 32 --langs $langs > $FILE_PREF

	cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp
	cat $FILE_PREF | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.ref
	sacrebleu -tok 'none' -s 'none' $FILE_PREF.ref < $FILE_PREF.hyp 2>&1 | tee ${RESULT_FILE}
	printf "CodeXGlue Evaluation: \t" >> ${RESULT_FILE}
	python evaluator.py $FILE_PREF.ref $FILE_PREF.hyp >> ${RESULT_FILE}
}


generate
