#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../..)

GPU=${1:-0}
LANGUAGE_GROUP=${2:-"one"}
LANG=${3:-"java"}
export CUDA_VISIBLE_DEVICES=$GPU

# Language Group must be one of the following
GROUP_LISTS="one compiled interpreted static dynamic strong weak all"
LANGUAGE_GROUP_CORRECT=$(echo $GROUP_LISTS | grep -w $LANGUAGE_GROUP)
if [[ $LANGUAGE_GROUP_CORRECT = "" ]]; then
    echo "LANGUAGE_GROUP(2nd parameter) must be one of the following"
    for group in $GROUP_LISTS; do
        printf "\t\"${group}\"\n"
    done
    exit
fi

GROUP_LISTS="java python go ruby php javascript"
LANGUAGE_GROUP_CORRECT=$(echo $GROUP_LISTS | grep -w $LANG)
if [[ $LANGUAGE_GROUP_CORRECT = "" ]]; then
    echo "LANGUAGE(3rd parameter) must be one of the following"
    for group in $GROUP_LISTS; do
        printf "\t\"${group}\"\n"
    done
    exit
fi

lang_dict="$HOME_DIR/multilingual/plbart/lang_dict.txt"
USER_DIR="$HOME_DIR/source"
PATH_2_DATA=${HOME_DIR}/multilingual/data/processed
PRETRAIN=${HOME_DIR}/multilingual/plbart/plbart_base_multilingual.pt
SAVE_DIR=${CURRENT_DIR}/${LANGUAGE_GROUP}
mkdir -p $SAVE_DIR

# we assume we will run this experiments in 1 GPU with bsz 32
BATCH_SIZE=8
UPDATE_FREQ=4

# CSNET data size is as follows
# java: 165k, python: 252k, php: 241k, go: 167k, javascript: 58k, ruby:25k
# So, number of mini-batches for each language would be:
# java: ~5100, python: ~7800, php: ~7500, go: ~5200, javascript: ~1800, ruby: ~780

declare -A LANG_WISE_WARMUP
LANG_WISE_WARMUP['java']=5000
LANG_WISE_WARMUP['python']=5000
LANG_WISE_WARMUP['php']=5000
LANG_WISE_WARMUP['go']=5000
LANG_WISE_WARMUP['javascript']=2000
LANG_WISE_WARMUP['ruby']=1000

function train() {

    fairseq-train "$PATH_2_DATA/binary" \
        --fp16 \
        --user-dir $USER_DIR \
        --restore-file $PRETRAIN \
        --reset-dataloader \
        --reset-optimizer \
        --reset-meters \
        --reset-lr-scheduler \
        --task translation_multi_simple_epoch_extended \
        --lang-dict $lang_dict \
        --lang-pairs $lang_pairs \
        --batch-size $BATCH_SIZE \
        --update-freq $UPDATE_FREQ \
        --arch mbart_base \
        --sampling-method "temperature" \
        --sampling-temperature 1.5 \
        --encoder-langtok "src" \
        --decoder-langtok \
        --truncate-source \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --optimizer adam \
        --adam-eps 1e-06 \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay \
        --lr 5e-05 \
        --warmup-updates $WARMUP \
        --max-update $MAX_UPDATE \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --weight-decay 0.1 \
        --eval-bleu \
        --eval-bleu-detok space \
        --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-args '{"beam": 1}' \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --skip-invalid-size-inputs-valid-test \
        --no-epoch-checkpoints \
        --patience 5 \
        --seed 1234 \
        --log-format json \
        --log-interval 100 \
        --save-dir $SAVE_DIR \
        --valid-subset valid \
        2>&1 | tee $SAVE_DIR/train.log
}

function evaluate() {

    TARGET_LANG=$1
    model=${SAVE_DIR}/checkpoint_best.pt
    FILE_PREF=${SAVE_DIR}/${TARGET_LANG}_output
    RESULT_FILE=${SAVE_DIR}/${TARGET_LANG}_result.txt
    GOUND_TRUTH_PATH="$PATH_2_DATA/test.${TARGET_LANG}-en_XX.${TARGET_LANG}"

    fairseq-generate "$PATH_2_DATA/binary" \
        --user-dir $USER_DIR \
        --path $model \
        --task translation_multi_simple_epoch_extended \
        --gen-subset test \
        --source-lang en_XX \
        --target-lang $TARGET_LANG \
        --sacrebleu \
        --remove-bpe 'sentencepiece' \
        --batch-size 16 \
        --encoder-langtok "src" \
        --decoder-langtok \
        --lang-dict $lang_dict \
        --lang-pairs $lang_pairs \
        --beam 5 >${FILE_PREF}

    cat $FILE_PREF | grep -P "^H" | sort -V | cut -f 3- | cut -d' ' -f 2- >$FILE_PREF.hyp
    if [[ "$(wc -l <${FILE_PREF}.hyp)" -eq "$(wc -l <$GOUND_TRUTH_PATH)" ]]; then
        export PYTHONPATH=${HOME_DIR}
        python ${HOME_DIR}/evaluation/pl_eval.py \
            --references ${GOUND_TRUTH_PATH} \
            --predictions ${FILE_PREF}.hyp \
            --detokenize \
            --lang $TARGET_LANG 2>&1 | tee ${RESULT_FILE}
    else
        echo 'Warning: Number of predictions do not match the number of ground truth!' >${RESULT_FILE}
    fi

}

if [[ "$LANGUAGE_GROUP" == 'one' ]]; then
    lang_pairs="en_XX-$LANG"
    MAX_UPDATE=100000
    WARMUP=${LANG_WISE_WARMUP[${LANG}]}
    SAVE_DIR=${SAVE_DIR}/en_${LANG}
    mkdir -p $SAVE_DIR
    train
    evaluate $LANG
else
    MAX_UPDATE=200000
    WARMUP=5000
    SAVE_DIR=${SAVE_DIR}/generation
    mkdir -p $SAVE_DIR
    if [[ "$LANGUAGE_GROUP" == 'all' ]]; then
        languages=(java python javascript php ruby go)
    elif [[ "$LANGUAGE_GROUP" == 'compiled' ]]; then
        languages=(java ruby go)
    elif [[ "$LANGUAGE_GROUP" == 'interpreted' ]]; then
        languages=(php python javascript)
    elif [[ "$LANGUAGE_GROUP" == 'static' ]]; then
        languages=(java go)
    elif [[ "$LANGUAGE_GROUP" == 'dynamic' ]]; then
        languages=(javascript python php ruby)
    elif [[ "$LANGUAGE_GROUP" == 'strong' ]]; then
        languages=(java go python ruby)
    elif [[ "$LANGUAGE_GROUP" == 'weak' ]]; then
        languages=(php javascript)
    fi

    # a list language pairs to train multilingual models, e.g. "en-java,en-python"
    lang_pairs=""
    for lang in ${languages[*]}; do
        lang_pairs=$lang_pairs"en_XX-$lang,"
    done
    lang_pairs=${lang_pairs::-1}

    train
    for lang in ${languages[*]}; do
        evaluate $lang
    done
fi
