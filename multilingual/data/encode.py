#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from tqdm import tqdm

from multiprocessing import Pool
import sentencepiece as spm


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global sp
        sp = spm.SentencePieceProcessor(model_file=self.args.model_file)

    def _encode(self, line):
        global sp
        return sp.encode(line, out_type=str)

    def _decode(self, tokens):
        global sp
        return sp.decode(tokens)

    def encode(self, input):
        assert isinstance(input, list)
        assert len(input) == 2
        source, target = input[0], input[1]
        if len(source) == 0 and not self.args.keep_empty:
            return ["EMPTY", None, None]
        if len(target) == 0 and not self.args.keep_empty:
            return ["EMPTY", None, None]
        src_tokens = self._encode(source)[:self.args.max_len]
        tgt_tokens = self._encode(target)[:self.args.max_len]
        source, target = " ".join(src_tokens), " ".join(tgt_tokens)
        return ["PASS", source, target]


def process(args):
    data = []
    with open(args.input_source, 'r', encoding='utf-8') as f1, \
            open(args.input_target, 'r', encoding='utf-8') as f2:
        for src, tgt in zip(f1, f2):
            data.append([src, tgt])

    encoder = MultiprocessingEncoder(args)
    pool = Pool(args.workers, initializer=encoder.initializer)

    processed_dataset = []
    with tqdm(total=len(data), desc='Processing') as pbar:
        for i, ex in enumerate(pool.imap(encoder.encode, data, 100)):
            pbar.update()
            processed_dataset.append(ex)

    filtered = 0
    with open(args.output_source, 'w', encoding='utf-8') as fw1, \
            open(args.output_target, 'w', encoding='utf-8') as fw2:
        for (filt, source, target) in processed_dataset:
            if filt == "PASS":
                fw1.write(source + '\n')
                fw2.write(target + '\n')
            else:
                filtered += 1

    if filtered > 0:
        print("filtered {} lines".format(filtered), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", help='path to *.model file')
    parser.add_argument("--input_source", type=str, required=True, help="input source file")
    parser.add_argument("--input_target", type=str, required=True, help="input target file")
    parser.add_argument("--output_source", type=str, required=True, help="output source file")
    parser.add_argument("--output_target", type=str, required=True, help="output target file")
    parser.add_argument("--keep_empty", action="store_true", help="keep empty lines")
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
