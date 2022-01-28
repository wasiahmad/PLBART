#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../../../../..)

DATA_DIR=${CURRENT_DIR}/python
DOWNLOAD_DIR=${DATA_DIR}/download
SPM_PROC_DATA_DIR=${DATA_DIR}/spm
SHARD_DIR=${DATA_DIR}/shards
SPM_MODEL_DIR=${HOME_DIR}/sentencepiece
SPM_ENC_SCRIPT=${SPM_MODEL_DIR}/encode.py
SPM_DICT_FILE=${SPM_MODEL_DIR}/dict.txt

function download() {

    URL_PREFIX="https://huggingface.co/datasets/lvwerra/codeparrot-clean/resolve/main"
    mkdir -p $DOWNLOAD_DIR
    for number in {1..54}; do
        file_id=$(printf "%012d" $number)
        filename=file-${file_id}.json.gz
        file_url=${URL_PREFIX}/$filename
        echo "Downloading $file_url"
        wget $file_url -P $DOWNLOAD_DIR
    done

}

function process_and_extract() {

    DEST_DIR=${HOME_DIR}/data/github
    cd $DEST_DIR || exit
    python -m preprocessing.preprocess \
        $DOWNLOAD_DIR \
        --lang1 python \
        --keep_comments True \
        --test_size 10000
    cd $CURRENT_DIR || exit

}

function spm_preprocess() {

    mkdir -p $SPM_PROC_DATA_DIR
    for i in $(seq 0 7); do
        for j in functions_class functions_standalone; do
            echo "$PROCESSED_DATA_DIR/train.$i.$j.tok"
            python $SPM_ENC_SCRIPT \
                --model-file $SPM_MODEL_DIR/sentencepiece.bpe.model \
                --inputs $DOWNLOAD_DIR/train.with_comments.$i.$j.tok \
                --outputs $SPM_PROC_DATA_DIR/train.$i.$j.spm \
                --max_len 1020 \
                --workers 60
        done
        cat $SPM_PROC_DATA_DIR/train.$i.*.spm >$SPM_PROC_DATA_DIR/train.$i.functions.spm
        rm $SPM_PROC_DATA_DIR/train.$i.functions_*.spm
    done
    for SPLIT in valid test; do
        for j in functions_class functions_standalone; do
            python $SPM_ENC_SCRIPT \
                --model-file $SPM_MODEL_DIR/sentencepiece.bpe.model \
                --inputs $DOWNLOAD_DIR/$SPLIT.with_comments.$j.tok \
                --outputs $SPM_PROC_DATA_DIR/$SPLIT.$j.spm \
                --max_len 1020 \
                --workers 60
        done
        cat $SPM_PROC_DATA_DIR/$SPLIT.*.spm >$SPM_PROC_DATA_DIR/$SPLIT.functions.spm
        rm $SPM_PROC_DATA_DIR/$SPLIT.functions_*.spm
    done

}

function binarize() {

    mkdir -p $SHARD_DIR
    for i in $(seq 0 7); do
        mkdir -p $SHARD_DIR/shard${i}
        cp $SPM_DICT_FILE $SHARD_DIR/shard${i}
        fairseq-preprocess \
            --only-source \
            --trainpref $SPM_PROC_DATA_DIR/train.$i.functions.spm \
            --destdir $SHARD_DIR/shard${i} \
            --srcdict $SPM_DICT_FILE \
            --workers 60
    done
    fairseq-preprocess \
        --only-source \
        --validpref $SPM_PROC_DATA_DIR/valid.functions.spm \
        --testpref $SPM_PROC_DATA_DIR/test.functions.spm \
        --destdir $SHARD_DIR/shard0 \
        --srcdict $SPM_DICT_FILE \
        --workers 60

}

download
process_and_extract
spm_preprocess
binarize
