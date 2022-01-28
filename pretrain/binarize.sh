#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../)

SPM_DIR=${HOME_DIR}/sentencepiece
DICT_FILE=${SPM_DIR}/dict.txt
SPM_VOCAB=${SPM_DIR}/sentencepiece.bpe.vocab
SPM_ENC_SCRIPT=${SPM_DIR}/encode.py

DATA_DIR=${HOME_DIR}/data
GITHUB_DIR=${DATA_DIR}/github
SO_DIR=${DATA_DIR}/stackoverflow/desc_shards

SHARD_DIR=${DATA_DIR}/shards
mkdir -p $SHARD_DIR

function spm_preprocess() {

    for LANG in java python; do
        for i in $(seq 0 7); do
            for j in functions_class functions_standalone; do
                echo "$GITHUB_DIR/$LANG/train.$i.$j.tok"
                python $SPM_ENC_SCRIPT \
                    --model-file $SPM_DIR/sentencepiece.bpe.model \
                    --inputs "$GITHUB_DIR/$LANG/train.$i.$j.tok" \
                    --outputs "$GITHUB_DIR/$LANG/train.$i.$j.spm" \
                    --max_len 510 \
                    --workers 60
            done
            cat $GITHUB_DIR/$LANG/train.$i.*.spm >$GITHUB_DIR/$LANG/train.$i.functions.spm
            rm $GITHUB_DIR/$LANG/train.$i.functions_class.spm
            rm $GITHUB_DIR/$LANG/train.$i.functions_standalone.spm
        done
        for SPLIT in valid test; do
            for j in functions_class functions_standalone; do
                python $SPM_ENC_SCRIPT \
                    --model-file $SPM_DIR/sentencepiece.bpe.model \
                    --inputs "$GITHUB_DIR/$LANG/$SPLIT.$j.tok" \
                    --outputs "$GITHUB_DIR/$LANG/$SPLIT.$j.spm" \
                    --max_len 510 \
                    --workers 60
            done
            cat $GITHUB_DIR/$LANG/$SPLIT.*.spm >$GITHUB_DIR/$LANG/$SPLIT.functions.spm
            rm $GITHUB_DIR/$LANG/$SPLIT.functions_class.spm
            rm $GITHUB_DIR/$LANG/$SPLIT.functions_standalone.spm
        done
    done

}

function spm_so_preprocess() {

    for LANG in java python; do
        for i in $(seq 0 7); do
            echo "$SO_DIR/train.$i.descriptions.txt"
            python $SPM_ENC_SCRIPT \
                --model-file $SPM_DIR/sentencepiece.bpe.model \
                --inputs "$SO_DIR/train.$i.description.txt" \
                --outputs "$SO_DIR/train.$i.description.spm" \
                --max_len 510 \
                --workers 60
        done
        for SPLIT in valid test; do
            python $SPM_ENC_SCRIPT \
                --model-file $SPM_DIR/sentencepiece.bpe.model \
                --inputs "$SO_DIR/$SPLIT.description.txt" \
                --outputs "$SO_DIR/$SPLIT.description.spm" \
                --max_len 510 \
                --workers 60
        done
    done

}

function binarize() {

    for LANG in java python; do
        for i in $(seq 0 7); do
            mkdir -p $SHARD_DIR/shard${i}
            cp $DICT_FILE $SHARD_DIR/shard${i}
            fairseq-preprocess \
                --only-source \
                --trainpref $GITHUB_DIR/$LANG/train.$i.functions.spm \
                --destdir $SHARD_DIR/shard${i}/$LANG \
                --srcdict $DICT_FILE \
                --workers 60
        done
        fairseq-preprocess \
            --only-source \
            --validpref $GITHUB_DIR/$LANG/valid.functions.spm \
            --testpref $GITHUB_DIR/$LANG/test.functions.spm \
            --destdir $SHARD_DIR/shard0/$LANG \
            --srcdict $DICT_FILE \
            --workers 60
    done

}

function so_binarize() {

    for i in $(seq 0 7); do
        mkdir -p $SHARD_DIR/shard${i}
        cp $DICT_FILE $SHARD_DIR/shard${i}
        fairseq-preprocess \
            --only-source \
            --trainpref $SO_DIR/train.$i.description.spm \
            --destdir $SHARD_DIR/shard${i}/en_XX \
            --srcdict $DICT_FILE \
            --workers 60
    done
    fairseq-preprocess \
        --only-source \
        --validpref $SO_DIR/valid.description.spm \
        --testpref $SO_DIR/test.description.spm \
        --destdir $SHARD_DIR/shard0/en_XX \
        --srcdict $DICT_FILE \
        --workers 60

}

if [[ ! -f $DICT_FILE ]]; then
    cut -f1 $SPM_VOCAB | tail -n +4 | sed "s/$/ 100/g" >$DICT_FILE
    for ((idx = 0; idx <= 7; idx++)); do
        mkdir -p ${SHARD_DIR}/shard${idx}
        cp $DICT_FILE ${SHARD_DIR}/shard${idx}
    done
fi

spm_preprocess
spm_so_preprocess
binarize
so_binarize
