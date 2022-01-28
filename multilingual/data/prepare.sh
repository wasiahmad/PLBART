#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../..)

DATA_DIR=${HOME_DIR}/multilingual/data/processed
SPM_DIR=${HOME_DIR}/sentencepiece
LANG_EN=en_XX

function spm_preprocess() {

    LANG=$1
    for SPLIT in train valid test; do
        if [[ ! -f $DATA_DIR/${SPLIT}.$LANG-$LANG_EN.spm.$LANG ]]; then
            python encode.py \
                --model_file ${SPM_DIR}/sentencepiece.bpe.model \
                --input_source $DATA_DIR/${SPLIT}.$LANG-$LANG_EN.$LANG \
                --input_target $DATA_DIR/${SPLIT}.$LANG-$LANG_EN.$LANG_EN \
                --output_source $DATA_DIR/${SPLIT}.$LANG-$LANG_EN.spm.$LANG \
                --output_target $DATA_DIR/${SPLIT}.$LANG-$LANG_EN.spm.$LANG_EN \
                --max_len 510 \
                --workers 60
        fi
    done

}

function binarize() {

    LANG=$1

    fairseq-preprocess \
        --source-lang $LANG \
        --target-lang $LANG_EN \
        --trainpref $DATA_DIR/train.$LANG-$LANG_EN.spm \
        --validpref $DATA_DIR/valid.$LANG-$LANG_EN.spm \
        --testpref $DATA_DIR/test.$LANG-$LANG_EN.spm \
        --destdir $DATA_DIR/binary \
        --workers 60 \
        --srcdict ${SPM_DIR}/dict.txt \
        --tgtdict ${SPM_DIR}/dict.txt

    fairseq-preprocess \
        --source-lang $LANG_EN \
        --target-lang $LANG \
        --trainpref $DATA_DIR/train.$LANG-$LANG_EN.spm \
        --validpref $DATA_DIR/valid.$LANG-$LANG_EN.spm \
        --testpref $DATA_DIR/test.$LANG-$LANG_EN.spm \
        --destdir $DATA_DIR/binary \
        --workers 60 \
        --srcdict ${SPM_DIR}/dict.txt \
        --tgtdict ${SPM_DIR}/dict.txt

}

mkdir -p $DATA_DIR
PYTHONPATH=${HOME_DIR} python process.py
for lang in java python ruby go javascript php; do
    spm_preprocess $lang && binarize $lang
done
