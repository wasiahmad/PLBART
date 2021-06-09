import os
import json
import argparse
import subprocess
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


class MultiprocessingEncoder(object):

    def __init__(self, model_file, max_length):
        self.model_file = model_file
        self.max_length = max_length

    def initializer(self):
        global sp
        sp = spm.SentencePieceProcessor(model_file=self.model_file)

    def _encode(self, line):
        global sp
        return sp.encode(line, out_type=str)

    def _decode(self, tokens):
        global sp
        return sp.decode(tokens)

    def encode(self, example):
        code_tokens = self._encode(example[0])
        return {
            'code': " ".join(code_tokens)[:self.max_length],
            'label': example[1]
        }


def process(
        spmfile, srcfile, index_file, outdir, split,
        nexample=-1, max_length=510, workers=20
):
    index = set()
    with open(index_file) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            index.add(int(line))

    js_all = json.load(open(srcfile))
    data = []
    for idx, js in enumerate(js_all):
        if idx in index:
            data.append((js['func'], js['target']))

    encoder = MultiprocessingEncoder(spmfile, max_length)
    pool = Pool(workers, initializer=encoder.initializer)

    processed_dataset = []
    with tqdm(total=len(data), desc='Processing') as pbar:
        for i, ex in enumerate(pool.imap(encoder.encode, data, 100)):
            pbar.update()
            processed_dataset.append(ex)

    with open(os.path.join(outdir, '{}.input0'.format(split)), 'w', encoding='utf-8') as fw1, \
            open(os.path.join(outdir, '{}.label'.format(split)), 'w', encoding='utf-8') as fw2:
        for idx, ex in enumerate(processed_dataset):
            if nexample != -1 and idx >= nexample:
                break
            if ex['label'] not in [0, 1]:
                continue
            fw1.write(ex['code'] + '\n')
            fw2.write(str(ex['label']) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        help='path to *.model file',
    )
    parser.add_argument(
        "--src_file",
        type=str,
        help="input files (.jsonl) to filter/encode",
    )
    parser.add_argument(
        "--tgt_file",
        type=str,
        help="output file (.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path of the output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default='train',
        help="path of the output directory",
    )
    parser.add_argument(
        "--nexample",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=510
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default='-',
        help="path of the index file",
    )
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    process(
        args.model_file, args.src_file, args.index_file,
        args.output_dir, args.split, args.nexample,
        args.max_length, args.workers
    )


if __name__ == "__main__":
    main()
