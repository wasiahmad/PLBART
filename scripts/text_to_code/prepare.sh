#!/usr/bin/env bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

DATA_DIR=${HOME_DIR}/data/codeXglue/text-to-code
SPM_DIR=${HOME_DIR}/sentencepice


function spm_preprocess () {

dataset=$1
tgt_lang=$2

for SPLIT in train valid test; do
    if [[ ! -f $DATA_DIR/$dataset/${SPLIT}.spm.$tgt_lang ]]; then
        if [[ $SPLIT == 'test' ]]; then
            MAX_LEN=9999 # we do not truncate test sequences
        else
            MAX_LEN=510
        fi
        python encode.py \
            --model-file ${SPM_DIR}/sentencepiece.bpe.model \
            --input_file $DATA_DIR/$dataset/${SPLIT}.json \
            --output_dir $DATA_DIR/$dataset \
            --src_field nl \
            --tgt_field code \
            --src_lang en_XX \
            --tgt_lang $tgt_lang \
            --pref $SPLIT \
            --max_len $MAX_LEN \
            --workers 60;
    fi
done

}


function binarize () {

dataset=$1
tgt_lang=$2

if [[ ! -d $DATA_DIR/$dataset/data-bin ]]; then
    fairseq-preprocess \
        --source-lang en_XX \
        --target-lang $tgt_lang \
        --trainpref $DATA_DIR/$dataset/train.spm \
        --validpref $DATA_DIR/$dataset/valid.spm \
        --testpref $DATA_DIR/$dataset/test.spm \
        --destdir $DATA_DIR/$dataset/data-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --workers 60 \
        --srcdict ${SPM_DIR}/dict.txt \
        --tgtdict ${SPM_DIR}/dict.txt;
fi

}

spm_preprocess concode java
binarize concode java
