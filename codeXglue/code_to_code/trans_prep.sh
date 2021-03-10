#!/usr/bin/env bash

SRCDIR=/home/username/workspace/projects/CodeBART/data/codeXglue/code-to-code/translation
SPMDIR=/local/username/codebart

function spm_preprocess () {

for SPLIT in train valid test; do
    if [[ $SPLIT == 'test' ]]; then
        MAX_LEN=9999 # we do not truncate test sequences
    else
        MAX_LEN=510
    fi
    python encode.py \
        --model-file ${SPMDIR}/sentencepiece.bpe.model \
        --src_file $SRCDIR/${SPLIT}.java-cs.txt.java \
        --tgt_file $SRCDIR/${SPLIT}.java-cs.txt.cs \
        --output_dir $SRCDIR \
        --src_lang java --tgt_lang cs \
        --pref $SPLIT --max_len $MAX_LEN \
        --workers 60;
done

}

function binarize () {

fairseq-preprocess \
    --source-lang java \
    --target-lang cs \
    --trainpref $SRCDIR/train.spm \
    --validpref $SRCDIR/valid.spm \
    --testpref $SRCDIR/test.spm \
    --destdir $SRCDIR/data-bin \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 60 \
    --srcdict ${SPMDIR}/dict.txt \
    --tgtdict ${SPMDIR}/dict.txt;

}

spm_preprocess
binarize
