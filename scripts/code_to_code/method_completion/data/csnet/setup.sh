#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../../../../..)

DATA_DIR=${CURRENT_DIR}/python
DOWNLOAD_DIR=${DATA_DIR}/download
SPM_MODEL_DIR=${HOME_DIR}/sentencepiece
SPM_ENC_SCRIPT=${SPM_MODEL_DIR}/encode.py
SPM_DICT_FILE=${SPM_MODEL_DIR}/dict.txt
SPM_PROC_DATA_DIR=${DATA_DIR}/spm

function download() {

    URL_PREFIX=https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2
    if [[ ! -d $DOWNLOAD_DIR ]]; then
        mkdir -p $DOWNLOAD_DIR
        wget $URL_PREFIX/python.zip -O python.zip
        unzip python.zip -d $CURRENT_DIR && rm python.zip
        rm $CURRENT_DIR/python_licenses.pkl
        mv $CURRENT_DIR/python_dedupe_definitions_v2.pkl $DOWNLOAD_DIR
        mv $CURRENT_DIR/python/final/*/*/* $DOWNLOAD_DIR
        rm -rf $CURRENT_DIR/python/final
        cd $DOWNLOAD_DIR || exit
        for file in ./*.jsonl.gz; do
            gzip -d $file
        done
        cd $CURRENT_DIR || exit
    fi

}

function prepare() {

    for split in train valid test; do
        if [[ ! -f $DOWNLOAD_DIR/${split}.functions.tok ]]; then
            export PYTHONPATH=$HOME_DIR
            python preprocess.py \
                --split $split \
                --source_dir $DOWNLOAD_DIR \
                --target_dir $DOWNLOAD_DIR \
                --workers 60
        fi
    done

}

function spm_preprocess() {

    mkdir -p $SPM_PROC_DATA_DIR
    for SPLIT in train valid test; do
        python $SPM_ENC_SCRIPT \
            --model-file $SPM_MODEL_DIR/sentencepiece.bpe.model \
            --inputs $DATA_DIR/$SPLIT.functions.tok \
            --outputs $SPM_PROC_DATA_DIR/$SPLIT.functions.spm \
            --max_len 1020 \
            --workers 60
    done
}

function binarize() {

    fairseq-preprocess \
        --only-source \
        --trainpref $SPM_PROC_DATA_DIR/train.functions.spm \
        --validpref $SPM_PROC_DATA_DIR/valid.functions.spm \
        --testpref $SPM_PROC_DATA_DIR/test.functions.spm \
        --destdir $DATA_DIR/data-bin \
        --srcdict $SPM_DICT_FILE \
        --workers 60

}

download
prepare
spm_preprocess
binarize
