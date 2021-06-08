#!/usr/bin/env bash

#############################################
#        Download PLBART checkpoints        #
#############################################

FILE=checkpoint_11_100000.pt
# https://drive.google.com/file/d/19OLKx0YY0yVorzZa-caFW0-hALVvX7gt

if [[ ! -f "$FILE" ]]; then
    fileid="19OLKx0YY0yVorzZa-caFW0-hALVvX7gt"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
fi
