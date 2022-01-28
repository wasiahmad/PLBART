#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../..)

DATA_DIR=${HOME_DIR}/data/codeXglue/code-to-text
SPM_DIR=${HOME_DIR}/sentencepiece

function spm_preprocess() {

    LANG=$1
    for SPLIT in train valid test; do
        if [[ ! -f $DATA_DIR/$LANG/${SPLIT}.spm.$LANG ]]; then
            if [[ $SPLIT == 'test' ]]; then
                MAX_TGT_LEN=9999 # we do not truncate test sequences
            else
                MAX_TGT_LEN=128
            fi
            if [[ $LANG == 'python' ]]; then
                SRC_FIELD='code'
            else
                SRC_FIELD='code_tokens'
            fi
            python encode.py \
                --model-file ${SPM_DIR}/sentencepiece.bpe.model \
                --input_file $DATA_DIR/$LANG/${SPLIT}.jsonl \
                --output_dir $DATA_DIR/$LANG \
                --src_field $SRC_FIELD \
                --tgt_field docstring_tokens \
                --src_lang $LANG \
                --tgt_lang en_XX \
                --pref $SPLIT \
                --max_src_len 256 \
                --max_tgt_len $MAX_TGT_LEN \
                --workers 60
        fi
    done

}

function binarize() {

    LANG=$1
    if [[ ! -d $DATA_DIR/$LANG/data-bin ]]; then
        fairseq-preprocess \
            --source-lang $LANG \
            --target-lang en_XX \
            --trainpref $DATA_DIR/$LANG/train.spm \
            --validpref $DATA_DIR/$LANG/valid.spm \
            --testpref $DATA_DIR/$LANG/test.spm \
            --destdir $DATA_DIR/$LANG/data-bin \
            --thresholdtgt 0 \
            --thresholdsrc 0 \
            --workers 60 \
            --srcdict ${SPM_DIR}/dict.txt \
            --tgtdict ${SPM_DIR}/dict.txt
    fi

}

for lang in java python ruby go javascript php; do
    spm_preprocess $lang
    binarize $lang
done
