#!/usr/bin/env bash

BASE_DIR=/local/username/codebart
HOME_DIR=/home/username/workspace/projects/CodeBART
langs=java,python,en_XX

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1

#checkpoint_3_25000.pt  checkpoint_6_50000.pt  checkpoint_8_75000.pt checkpoint_11_100000.pt
PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt
PRETRAIN=${BASE_DIR}/checkpoints/${PRETRAINED_MODEL_NAME}

PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-code/clone_detection/processed/data-bin

SAVE_DIR=${BASE_DIR}/code-to-code/clone_detection/${PRETRAINED_MODEL_NAME}-max-epoch_3
mkdir -p ${SAVE_DIR}
USER_DIR=${HOME_DIR}/src

function fine_tune () {
    OUTPUT_FILE=${SAVE_DIR}/finetune.log

    MAX_UPDATES=62500       # 10 epochs through 100k examples with bsz 16
    WARMUP_UPDATES=1000     # 6 percent of the number of updates
    LR=5e-5                 # Peak LR for polynomial LR scheduler.
    NUM_CLASSES=2
    MAX_SENTENCES=4         # Batch size.
    UPDATE_FREQ=4

    fairseq-train $PATH_2_DATA \
        --user-dir $USER_DIR \
        --restore-file $PRETRAIN \
        --max-positions 512 --langs $langs \
        --task bart_sentence_prediction --add-prev-output-tokens \
        --shorten-method "truncate" \
        --layernorm-embedding --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 1 \
        --init-token 0 \
        --arch mbart_base --criterion sentence_prediction \
        --num-classes $NUM_CLASSES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
        --clip-norm 1.0 --lr-scheduler polynomial_decay --lr $LR \
        --warmup-updates $WARMUP_UPDATES --update-freq $UPDATE_FREQ \
        --batch-size $MAX_SENTENCES --max-epoch 3 --max-update $MAX_UPDATES \
        --max-tokens 2048 \
        --seed 1234 --log-format json --log-interval 10 \
        --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints --find-unused-parameters \
        --ddp-backend no_c10d --save-dir $SAVE_DIR 2>&1 | tee ${OUTPUT_FILE}
}


function generate () {
	model=${SAVE_DIR}/checkpoint_best.pt
	RESULT_FILE=${SAVE_DIR}/result.txt

	python eval.py \
	    --user_dir $USER_DIR \
        --model_dir ${SAVE_DIR} \
        --model_name checkpoint_best.pt \
        --data_bin_path ${PATH_2_DATA} \
        --input_file ${HOME_DIR}/data/codeXglue/code-to-code/clone_detection/processed/test.input0 \
        --label_file ${HOME_DIR}/data/codeXglue/code-to-code/clone_detection/processed/test.label \
        --batch_size 128 \
        --max_example -1 \
        --output $RESULT_FILE;
}

#fine_tune
generate
