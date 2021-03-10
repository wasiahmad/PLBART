#!/usr/bin/env bash

SRCDIR=/home/username/workspace/projects/CodeBART/data/codeXglue/code-to-text
SPMDIR=/local/username/codebart

function spm_preprocess () {

LANG=$1
for SPLIT in train valid test; do
    if [[ ! -f $SRCDIR/$LANG/${SPLIT}.spm.$LANG ]]; then
        if [[ $SPLIT == 'test' ]]; then
            MAX_TGT_LEN=9999 # we do not truncate test sequences
        else
            MAX_TGT_LEN=128
        fi
        if [[ $LANG == 'python' ]]; then
            SRC_FIELD='code' # we do not truncate test sequences
        else
            SRC_FIELD='code_tokens'
        fi
        python encode.py \
            --model-file ${SPMDIR}/sentencepiece.bpe.model \
            --input_file $SRCDIR/$LANG/${SPLIT}.jsonl \
            --output_dir $SRCDIR/$LANG \
            --src_field $SRC_FIELD \
            --tgt_field docstring_tokens \
            --src_lang $LANG \
            --tgt_lang en_XX \
            --pref $SPLIT \
            --max_src_len 256 \
            --max_tgt_len $MAX_TGT_LEN \
            --workers 60;
    fi
done

}

function binarize () {

LANG=$1
if [[ ! -d $SRCDIR/$LANG/data-bin ]]; then
    fairseq-preprocess \
        --source-lang $LANG \
        --target-lang en_XX \
        --trainpref $SRCDIR/$LANG/train.spm \
        --validpref $SRCDIR/$LANG/valid.spm \
        --testpref $SRCDIR/$LANG/test.spm \
        --destdir $SRCDIR/$LANG/data-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --workers 60 \
        --srcdict ${SPMDIR}/dict.txt \
        --tgtdict ${SPMDIR}/dict.txt;
fi

}

for lang in java python ruby go javascript php; do
    spm_preprocess $lang
    binarize $lang
done
