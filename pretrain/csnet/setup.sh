#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../..)

OUT_DIR=${CURRENT_DIR}/data
mkdir -p $OUT_DIR

function download() {

    URL_PREFIX=https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2
    for lang in java python javascript php go ruby; do
        if [[ ! -d $OUT_DIR/$lang ]]; then
            wget $URL_PREFIX/${lang}.zip -O ${lang}.zip
            unzip ${lang}.zip -d $OUT_DIR
            rm ${lang}.zip
            rm $OUT_DIR/${lang}_licenses.pkl
            mv $OUT_DIR/${lang}_dedupe_definitions_v2.pkl $OUT_DIR/$lang
            mv $OUT_DIR/$lang/final/*/*/* $OUT_DIR/$lang
            rm -rf $OUT_DIR/$lang/final
            cd $OUT_DIR/$lang
            for file in ./*.jsonl.gz; do
                gzip -d $file
            done
            cd $CURRENT_DIR
        fi
    done

}

function prepare() {

    if [[ -d $OUT_DIR/en_XX ]]; then rm -rf $OUT_DIR/en_XX; fi
    mkdir -p $OUT_DIR/en_XX

    for lang in java python javascript php go ruby; do
        for split in train valid test; do
            if [[ ! -f $OUT_DIR/$lang/${split}.functions.tok ]]; then
                export PYTHONPATH=$HOME_DIR
                python preprocess.py \
                    --lang $lang \
                    --split $split \
                    --source_dir $OUT_DIR \
                    --target_pl_dir $OUT_DIR/$lang \
                    --target_nl_dir $OUT_DIR/en_XX \
                    --workers 60
            fi
        done
    done

}

function split() {

    NUM_SPLIT=$1
    for lang in java python javascript php go ruby; do
        PREFIX=$OUT_DIR/$lang/train
        if [[ -f $PREFIX.functions.tok ]]; then
            for ((i = 0; i < $NUM_SPLIT; i++)); do
                awk "NR % $NUM_SPLIT == $i" $PREFIX.functions.tok >$PREFIX.$i.functions.tok
            done
        fi
    done

    PREFIX=$OUT_DIR/en_XX/train
    if [[ -f $PREFIX.docstring.tok ]]; then
        for ((i = 0; i < $NUM_SPLIT; i++)); do
            awk "NR % $NUM_SPLIT == $i" $PREFIX.docstring.tok >$PREFIX.$i.docstring.tok
        done
    fi

}

download
prepare
split 4
