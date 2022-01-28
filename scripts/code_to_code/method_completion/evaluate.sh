#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../../..)

while getopts ":h" option; do
    case $option in
    h | *) # display help
        echo
        echo "Syntax: bash run.sh GPU_ID EVAL_DATA DECODING_TYPE MODEL_SIZE"
        echo
        echo "EVAL_DATA choices: [human-eval|mbpp]"
        echo "DECODING_TYPE choices: [beam|sampling]"
        echo "MODEL_SIZE choices: [base|large]"
        echo
        exit
        ;;
    esac
done

GPU=$1
EVAL_DATA=${2:-"human-eval"}
DECODING_TYPE=${3:-"sampling"}
MODEL_SIZE=${4:-base}

export CUDA_VISIBLE_DEVICES=$GPU
PATH_2_DATA=${CURRENT_DIR}/data/github/python/shards/shard0
TASK="code_completion"
CKPT_DIR=${CURRENT_DIR}/plbart_${MODEL_SIZE}_github
CKPT_FILE=checkpoint_last.pt

DEC_PARAMS=("--beam 200 --nbest 100")
if [[ $DECODING_TYPE == 'beam' ]]; then
    :
elif [[ $DECODING_TYPE == 'sampling' ]]; then
    DEC_PARAMS+=("--sampling")
    DEC_PARAMS+=("--sampling_topp 0.95")
    DEC_PARAMS+=("--temperature 0.6")
else
    echo -n "... Wrong DECODING_TYPE choice!! available choices: [beam|sampling]"
    exit 1
fi

function human_eval() {

    PATH_2_EVAL_DATA=${CURRENT_DIR}/human-eval/data
    python ${CURRENT_DIR}/human-eval/decode.py \
        --checkpoint_dir $CKPT_DIR \
        --checkpoint_file $CKPT_FILE \
        --task $TASK \
        --data_dir $PATH_2_EVAL_DATA \
        --data_name_or_path $PATH_2_DATA \
        --output_file $CKPT_DIR/completion_human_eval.jsonl \
        --output_function_file $CKPT_DIR/predictions_human_eval.txt \
        --max_len_b 256 \
        --batch_size 8 \
        $(echo "${DEC_PARAMS[@]}") \
        2>&1 | tee "${CKPT_DIR}/log_human_eval.txt"

    evaluate_functional_correctness $CKPT_DIR/completion_human_eval.jsonl \
        --problem_file ${CURRENT_DIR}/human-eval/data/HumanEval.jsonl

}

function mbpp_eval() {

    PATH_2_EVAL_DATA=${CURRENT_DIR}/mbpp/data
    python ${CURRENT_DIR}/mbpp/decode.py \
        --checkpoint_dir $CKPT_DIR \
        --checkpoint_file $CKPT_FILE \
        --task $TASK \
        --data_dir $PATH_2_EVAL_DATA \
        --data_name_or_path $PATH_2_DATA \
        --output_file $CKPT_DIR/completion_mbpp.jsonl \
        --output_function_file $CKPT_DIR/predictions_mbpp.txt \
        --max_len_b 400 \
        --batch_size 8 \
        $(echo "${DEC_PARAMS[@]}") \
        2>&1 | tee "${CKPT_DIR}/log_mbpp.txt"

    evaluate_functional_correctness $CKPT_DIR/completion_mbpp.jsonl \
        --problem_file ${CURRENT_DIR}/mbpp/data/mbpp.jsonl

}

if [[ $EVAL_DATA == 'human-eval' ]]; then
    human_eval
elif [[ $EVAL_DATA == 'mbpp' ]]; then
    mbpp_eval
else
    echo -n "... Wrong EVAL_DATA choice!! available choices: [human-eval|mbpp]"
    exit 1
fi
