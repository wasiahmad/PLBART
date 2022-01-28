#!/usr/bin/env bash

CURRENT_DIR=$PWD
LIB=$CURRENT_DIR/third_party
mkdir -p $LIB

conda create --name plbart python=3.6.10
conda activate plbart
conda config --add channels conda-forge
conda config --add channels pytorch

conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
conda install submitit

cd $LIB || exit
# install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq || exit
git checkout 698e3b91ffa832c286c48035bdff78238b0de8ae
pip install .
cd ..
# install apex
git clone https://github.com/NVIDIA/apex
cd apex || exit
export CXX=g++
export CUDA_HOME=/usr/local/cuda-10.2
# https://github.com/NVIDIA/apex/issues/1043#issuecomment-773868833
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd $CURRENT_DIR || exit

pip install sacrebleu==1.2.11
pip install tree_sitter==0.2.1
pip install sentencepiece
