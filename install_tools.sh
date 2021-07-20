#!/usr/bin/env bash

set -eux  # for easier debugging

REPO=$PWD
LIB=$REPO/third_party
mkdir -p $LIB

# install conda env
conda create --name plbart --file conda-env.txt
conda init bash
conda activate plbart

pip install sacrebleu==1.2.11
pip install tree_sitter==0.2.1

# install fairseq
cd $LIB
rm -rf fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout 698e3b91ffa832c286c48035bdff78238b0de8ae
pip install .
cd $LIB
