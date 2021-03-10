#!/usr/bin/env bash

SRCDIR=/home/username/workspace/projects/CodeBART/data/codeXglue/code-to-code/clone_detection
SPMDIR=/local/username/codebart

function preprocess () {

python encode.py \
    --preprocess \
    --model_file ${SPMDIR}/sentencepiece.bpe.model \
    --src_file $SRCDIR/data.jsonl \
    --tgt_file $SRCDIR/data-processed.txt;

}

function spm_preprocess () {

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
        --src_file $SRCDIR/data-processed.txt \
        --tgt_file $SRCDIR/${SPLIT}.txt \
        --output_dir $SRCDIR/processed \
        --nexample $NEXAMPLE \
        --split $SPLIT;
done

}

function binarize () {

fairseq-preprocess \
    --only-source \
    --trainpref $SRCDIR/processed/train.input0 \
    --validpref $SRCDIR/processed/valid.input0 \
    --testpref $SRCDIR/processed/test.input0 \
    --destdir $SRCDIR/processed/data-bin/input0 \
    --workers 60 \
    --srcdict ${SPMDIR}/dict.txt;
fairseq-preprocess \
    --only-source \
    --trainpref $SRCDIR/processed/train.label \
    --validpref $SRCDIR/processed/valid.label \
    --destdir $SRCDIR/processed/data-bin/label \
    --workers 60;

}

#preprocess
spm_preprocess
binarize
