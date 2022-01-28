#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../../..)
DATA_DIR=${HOME_DIR}/data/codeXglue/code-to-code/clone_detection
SPM_DIR=${HOME_DIR}/sentencepiece

function preprocess() {

    python encode.py \
        --preprocess \
        --model_file ${SPM_DIR}/sentencepiece.bpe.model \
        --src_file $DATA_DIR/data.jsonl \
        --tgt_file $DATA_DIR/data-processed.txt \
        --workers 60

}

function spm_preprocess() {

    for SPLIT in train valid test; do
        if [[ $SPLIT == 'test' ]]; then
            NEXAMPLE=-1
        elif [[ $SPLIT == 'valid' ]]; then
            NEXAMPLE=10000
        else
            NEXAMPLE=100000
        fi
        python encode.py \
            --postprocess \
            --src_file $DATA_DIR/data-processed.txt \
            --index_file $DATA_DIR/${SPLIT}.txt \
            --output_dir $DATA_DIR/processed \
            --nexample $NEXAMPLE \
            --split $SPLIT
    done

}

function binarize() {

    fairseq-preprocess \
        --only-source \
        --trainpref $DATA_DIR/processed/train.input0 \
        --validpref $DATA_DIR/processed/valid.input0 \
        --testpref $DATA_DIR/processed/test.input0 \
        --destdir $DATA_DIR/processed/data-bin/input0 \
        --srcdict ${SPM_DIR}/dict.txt \
        --workers 60
    fairseq-preprocess \
        --only-source \
        --trainpref $DATA_DIR/processed/train.label \
        --validpref $DATA_DIR/processed/valid.label \
        --destdir $DATA_DIR/processed/data-bin/label \
        --workers 60

}

preprocess
spm_preprocess
binarize
