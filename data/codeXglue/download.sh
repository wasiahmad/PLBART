#!/usr/bin/env bash

#############################################
#        Download text-to-code data         #
#############################################

FILE=text-to-code.zip
# https://drive.google.com/file/d/1rQjQh4Mle3yYzQbn-CRs4L1moZaAqr90

if [[ ! -d "text-to-code" ]]; then
    fileid="1rQjQh4Mle3yYzQbn-CRs4L1moZaAqr90"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" >/dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi

#############################################
#        Download code-to-text data         #
#############################################

FILE=code-to-text.zip
# https://drive.google.com/file/d/1m1IvGgPhDBg-SL-LajtFGTLyAJVbD0i3

if [[ ! -d "code-to-text" ]]; then
    fileid="1m1IvGgPhDBg-SL-LajtFGTLyAJVbD0i3"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" >/dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi

#############################################
#        Download code-to-code data         #
#############################################

FILE=code-to-code.zip
# https://drive.google.com/file/d/15jokCxFQ9BUbptMsrfj4RdH_KiNkTRP2

if [[ ! -d "code-to-code" ]]; then
    fileid="15jokCxFQ9BUbptMsrfj4RdH_KiNkTRP2"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" >/dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileid}" -o ${FILE}
    rm ./cookie
    unzip ${FILE} && rm ${FILE}
fi
