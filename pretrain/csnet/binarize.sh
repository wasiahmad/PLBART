#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../..)

SPM_DIR=${HOME_DIR}/sentencepiece
DICT_FILE=${SPM_DIR}/dict.txt
SPM_VOCAB=${SPM_DIR}/sentencepiece.bpe.vocab
SPM_ENC_SCRIPT=${SPM_DIR}/encode.py

DATA_DIR=${CURRENT_DIR}/data
SHARD_DIR=${DATA_DIR}/shards
mkdir -p $SHARD_DIR

for ((idx = 0; idx <= 3; idx++)); do
    mkdir -p ${SHARD_DIR}/shard${idx}
    cp $DICT_FILE ${SHARD_DIR}/shard${idx}
done

function preprocess_pl() {

    for LANG in java python javascript php go ruby; do
        for i in $(seq 0 3); do
            python $SPM_ENC_SCRIPT \
                --model-file $SPM_DIR/sentencepiece.bpe.model \
                --inputs $DATA_DIR/$LANG/train.$i.functions.tok \
                --outputs $DATA_DIR/$LANG/train.$i.functions.spm \
                --max_len 510 \
                --workers 60
        done
        head -n 5000 $DATA_DIR/$LANG/valid.functions.tok >$DATA_DIR/$LANG/valid.0.functions.tok
        python $SPM_ENC_SCRIPT \
            --model-file $SPM_DIR/sentencepiece.bpe.model \
            --inputs $DATA_DIR/$LANG/valid.0.functions.tok \
            --outputs $DATA_DIR/$LANG/valid.functions.spm \
            --max_len 510 \
            --workers 60
    done

}

function preprocess_nl() {

    for i in $(seq 0 3); do
        python $SPM_ENC_SCRIPT \
            --model-file $SPM_DIR/sentencepiece.bpe.model \
            --inputs $DATA_DIR/en_XX/train.$i.docstring.tok \
            --outputs $DATA_DIR/en_XX/train.$i.docstring.spm \
            --max_len 510 \
            --workers 60
    done
    head -n 5000 $DATA_DIR/en_XX/valid.docstring.tok >$DATA_DIR/en_XX/valid.0.docstring.tok
    python $SPM_ENC_SCRIPT \
        --model-file $SPM_DIR/sentencepiece.bpe.model \
        --inputs $DATA_DIR/en_XX/valid.0.docstring.tok \
        --outputs $DATA_DIR/en_XX/valid.docstring.spm \
        --max_len 510 \
        --workers 60

}

function binarize_pl() {

    for LANG in java python javascript php go ruby; do
        for i in $(seq 0 3); do
            fairseq-preprocess \
                --only-source \
                --trainpref $DATA_DIR/$LANG/train.$i.functions.spm \
                --destdir $SHARD_DIR/shard${i}/$LANG \
                --srcdict $DICT_FILE \
                --workers 60
        done
        fairseq-preprocess \
            --only-source \
            --validpref $DATA_DIR/$LANG/valid.functions.spm \
            --destdir $SHARD_DIR/shard0/$LANG \
            --srcdict $DICT_FILE \
            --workers 60
    done

}

function binarize_nl() {

    for i in $(seq 0 3); do
        fairseq-preprocess \
            --only-source \
            --trainpref $DATA_DIR/en_XX/train.$i.docstring.spm \
            --destdir $SHARD_DIR/shard${i}/en_XX \
            --srcdict $DICT_FILE \
            --workers 60
    done
    fairseq-preprocess \
        --only-source \
        --validpref $DATA_DIR/en_XX/valid.docstring.spm \
        --destdir $SHARD_DIR/shard0/en_XX \
        --srcdict $DICT_FILE \
        --workers 60

}

preprocess_pl
preprocess_nl
binarize_pl
binarize_nl
