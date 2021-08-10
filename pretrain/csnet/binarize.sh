#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

SPM_DIR=${HOME_DIR}/sentencepiece
DICT_FILE=${SPM_DIR}/dict.txt
SPM_VOCAB=${SPM_DIR}/sentencepiece.bpe.vocab
SPM_ENC_SCRIPT=${SPM_DIR}/encode.py

DATA_DIR=${CURRENT_DIR}/data
SHARD_DIR=${DATA_DIR}/shard
mkdir -p $SHARD_DIR
cp $DICT_FILE $SHARD_DIR


function preprocess_pl () {

for LANG in java python javascript php go ruby; do
    python $SPM_ENC_SCRIPT \
        --model-file $SPM_DIR/sentencepiece.bpe.model \
        --inputs $DATA_DIR/$LANG/train.functions.tok \
        --outputs $DATA_DIR/$LANG/train.functions.spm \
        --max_len 510 \
        --workers 60;
    python $SPM_ENC_SCRIPT \
        --model-file $SPM_DIR/sentencepiece.bpe.model \
        --inputs $DATA_DIR/$LANG/valid.functions.tok \
        --outputs $DATA_DIR/$LANG/valid.functions.spm \
        --max_len 510 \
        --workers 60;
done

}

function preprocess_nl () {

python $SPM_ENC_SCRIPT \
    --model-file $SPM_DIR/sentencepiece.bpe.model \
    --inputs $DATA_DIR/en_XX/train.docstring.tok \
    --outputs $DATA_DIR/en_XX/train.docstring.spm \
    --max_len 510 \
    --workers 60;
python $SPM_ENC_SCRIPT \
    --model-file $SPM_DIR/sentencepiece.bpe.model \
    --inputs $DATA_DIR/en_XX/valid.docstring.tok \
    --outputs $DATA_DIR/en_XX/valid.docstring.spm \
    --max_len 510 \
    --workers 60;

}

function binarize_pl () {

for LANG in java python javascript php go ruby; do
    fairseq-preprocess \
        --only-source \
        --trainpref $DATA_DIR/$LANG/train.functions.spm \
        --validpref $DATA_DIR/$LANG/valid.functions.spm \
        --destdir $SHARD_DIR/$LANG \
        --srcdict $DICT_FILE \
        --workers 60;
done

}

function binarize_nl () {

fairseq-preprocess \
    --only-source \
    --trainpref $DATA_DIR/en_XX/train.docstring.spm \
    --validpref $DATA_DIR/en_XX/valid.docstring.spm \
    --destdir $SHARD_DIR/en_XX \
    --srcdict $DICT_FILE \
    --workers 60;

}

preprocess_pl
preprocess_nl
binarize_pl
binarize_nl
