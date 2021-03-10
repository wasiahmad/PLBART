#!/usr/bin/env bash

#pip install gdown
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
mv CodeSearchNet/* .
rm -rf CodeSearchNet
