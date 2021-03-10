#!/usr/bin/env bash

mkdir -p /local/username/
mkdir -p /local/username/codebart/
mkdir -p /local/username/codebart/nmt-finetuned/

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
            #echo "TGT_LANG  Language choices: [$(IFS=\| ; echo "${!LANG_MAP[@]}")]"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1
SRC_LANG=en
TGT_LANG=java

SOURCE=${LANG_MAP[$SRC_LANG]}
TARGET=${LANG_MAP[$TGT_LANG]}

PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt

PRETRAIN=/local/username/codebart/checkpoints/${PRETRAINED_MODEL_NAME}

PATH_2_DATA=/home/username/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/data-bin

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=/local/codebart/cbart/nmt-finetuned/${SOURCE}_${TARGET}-${PRETRAINED_MODEL_NAME}

FINETUNED_MODEL_NAME=$2

generate=TRUE
if [[ $# == 3 ]]; then
	generate=TRUE
else
	generate=FALSE
fi

function generate () {
	model=${SAVE_DIR}/${FINETUNED_MODEL_NAME}.pt
	FILE_PREF=${SAVE_DIR}/${FINETUNED_MODEL_NAME}-output
	mkdir -p ${FILE_PREF}
	RESULT_FILE=${FILE_PREF}/result.txt
	
	if [[ -f "$FILE_PREF/generated-output" ]]; then
		if [[ $generate == TRUE ]]; then
			fairseq-generate $PATH_2_DATA \
		                --path $model \
                		--task translation_from_pretrained_bart \
                		--gen-subset test \
                		-t $TARGET -s $SOURCE \
                		--sacrebleu --remove-bpe 'sentencepiece' \
                		--batch-size 8 --langs $langs --beam 10 > $FILE_PREF/generated-output;
		fi
	else
		fairseq-generate $PATH_2_DATA \
	                --path $model \
        	        --task translation_from_pretrained_bart \
                	--gen-subset test \
                	-t $TARGET -s $SOURCE \
                	--sacrebleu --remove-bpe 'sentencepiece' \
                	--batch-size 8 --langs $langs --beam 10 > $FILE_PREF/generated-output;
	fi
	cat $FILE_PREF/generated-output | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF/code.hyp
	cat $FILE_PREF/generated-output | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[${TARGET}\]//g' > $FILE_PREF/code.ref
	
	sacrebleu -tok 'none' -s 'none' $FILE_PREF/code.ref < $FILE_PREF/code.hyp 2>&1 | tee ${RESULT_FILE}
	printf "CodeXGlue Evaluation: \t" >> ${RESULT_FILE}
	python evaluator.py --expected ${FILE_PREF}/code.ref  --predicted ${FILE_PREF}/code.hyp
	python evaluator.py --expected ${FILE_PREF}/code.ref  --predicted ${FILE_PREF}/code.hyp >> ${RESULT_FILE}
	echo "CodeBLEU Evaluation" > ${RESULT_FILE}

	cd CodeBLEU;
	python calc_code_bleu.py --refs $FILE_PREF/code.ref --hyp $FILE_PREF/code.hyp --lang java >> ${RESULT_FILE};
	python calc_code_bleu.py --refs $FILE_PREF/code.ref --hyp $FILE_PREF/code.hyp --lang java;
	cd ..;
}

generate
