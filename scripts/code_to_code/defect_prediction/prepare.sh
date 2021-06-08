#!/usr/bin/env bash

CURRENT_DIR=`pwd`;
HOME_DIR=`realpath ../../..`;
DATA_DIR=${HOME_DIR}/data/codeXglue/code-to-code/defect_prediction;
SPM_DIR=${HOME_DIR}/sentencepiece;


function spm_preprocess () {

for SPLIT in train valid test; do
    python encode.py \
        --model_file ${SPM_DIR}/sentencepiece.bpe.model \
        --src_file $DATA_DIR/function.json \
        --index_file $DATA_DIR/${SPLIT}.txt \
        --output_dir $DATA_DIR/processed \
        --split $SPLIT;
done

}

function binarize () {

fairseq-preprocess \
    --only-source \
    --trainpref $DATA_DIR/processed/train.input0 \
    --validpref $DATA_DIR/processed/valid.input0 \
    --testpref $DATA_DIR/processed/test.input0 \
    --destdir $DATA_DIR/processed/data-bin/input0 \
    --workers 60 \
    --srcdict ${SPM_DIR}/dict.txt;
fairseq-preprocess \
    --only-source \
    --trainpref $DATA_DIR/processed/train.label \
    --validpref $DATA_DIR/processed/valid.label \
    --destdir $DATA_DIR/processed/data-bin/label \
    --workers 60;

}

spm_preprocess
binarize
