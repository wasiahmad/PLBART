#!/usr/bin/env bash

SRCDIR=/home/username/workspace/projects/CodeBART/data/codeXglue/text-to-code
SPMDIR=/local/username/codebart

function spm_preprocess () {

dataset=$1
tgt_lang=$2

for SPLIT in train valid test; do
    if [[ ! -f $SRCDIR/$dataset/${SPLIT}.spm.$tgt_lang ]]; then
        if [[ $SPLIT == 'test' ]]; then
            MAX_LEN=9999 # we do not truncate test sequences
        else
            MAX_LEN=510
        fi
        python encode.py \
            --model-file ${SPMDIR}/sentencepiece.bpe.model \
            --input_file $SRCDIR/$dataset/${SPLIT}.json \
            --output_dir $SRCDIR/$dataset \
            --src_field nl \
            --tgt_field code \
            --src_lang en_XX \
            --tgt_lang $tgt_lang \
            --pref $SPLIT \
            --max_len $MAX_LEN \
            --workers 60;
    fi
done

}

function binarize () {

dataset=$1
tgt_lang=$2

if [[ ! -d $SRCDIR/$dataset/data-bin ]]; then
    fairseq-preprocess \
        --source-lang en_XX \
        --target-lang $tgt_lang \
        --trainpref $SRCDIR/$dataset/train.spm \
        --validpref $SRCDIR/$dataset/valid.spm \
        --testpref $SRCDIR/$dataset/test.spm \
        --destdir $SRCDIR/$dataset/data-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --workers 60 \
        --srcdict ${SPMDIR}/dict.txt \
        --tgtdict ${SPMDIR}/dict.txt;
fi

}

spm_preprocess concode java
binarize concode java
spm_preprocess conala python
binarize conala python
spm_preprocess conala-mined python
binarize conala-mined python
