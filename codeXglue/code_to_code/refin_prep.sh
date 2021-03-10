#!/usr/bin/env bash

SRCDIR=/home/username/workspace/projects/CodeBART/data/codeXglue/code-to-code/refinement
SPMDIR=/local/username/codebart

function spm_preprocess () {

for LANG in small medium; do
    for SPLIT in train valid test; do
        if [[ $SPLIT == 'test' ]]; then
            MAX_LEN=9999 # we do not truncate test sequences
        else
            MAX_LEN=256
        fi
        python encode.py \
            --model-file ${SPMDIR}/sentencepiece.bpe.model \
            --src_file $SRCDIR/$LANG/${SPLIT}.buggy-fixed.buggy \
            --tgt_file $SRCDIR/$LANG/${SPLIT}.buggy-fixed.fixed \
            --output_dir $SRCDIR/$LANG \
            --src_lang source --tgt_lang target \
            --pref $SPLIT --max_len $MAX_LEN \
            --workers 60;
    done
done

}

function binarize () {

for LANG in small medium; do
    fairseq-preprocess \
        --source-lang source \
        --target-lang target \
        --trainpref $SRCDIR/$LANG/train.spm \
        --validpref $SRCDIR/$LANG/valid.spm \
        --testpref $SRCDIR/$LANG/test.spm \
        --destdir $SRCDIR/$LANG/data-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --workers 60 \
        --srcdict ${SPMDIR}/dict.txt \
        --tgtdict ${SPMDIR}/dict.txt;
done

}

spm_preprocess
binarize
