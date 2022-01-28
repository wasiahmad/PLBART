#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../../..)
DATA_DIR=${HOME_DIR}/data/codeXglue/code-to-code/refinement
SPM_DIR=${HOME_DIR}/sentencepiece

function spm_preprocess() {

    for LANG in small medium; do
        for SPLIT in train valid test; do
            if [[ $SPLIT == 'test' ]]; then
                MAX_LEN=9999 # we do not truncate test sequences
            else
                MAX_LEN=256
            fi
            python encode.py \
                --model-file ${SPM_DIR}/sentencepiece.bpe.model \
                --src_file $DATA_DIR/$LANG/${SPLIT}.buggy-fixed.buggy \
                --tgt_file $DATA_DIR/$LANG/${SPLIT}.buggy-fixed.fixed \
                --output_dir $DATA_DIR/$LANG \
                --src_lang source --tgt_lang target \
                --pref $SPLIT --max_len $MAX_LEN \
                --workers 60
        done
    done

}

function binarize() {

    for LANG in small medium; do
        fairseq-preprocess \
            --source-lang source \
            --target-lang target \
            --trainpref $DATA_DIR/$LANG/train.spm \
            --validpref $DATA_DIR/$LANG/valid.spm \
            --testpref $DATA_DIR/$LANG/test.spm \
            --destdir $DATA_DIR/$LANG/data-bin \
            --thresholdtgt 0 \
            --thresholdsrc 0 \
            --workers 60 \
            --srcdict ${SPM_DIR}/dict.txt \
            --tgtdict ${SPM_DIR}/dict.txt
    done

}

spm_preprocess
binarize
