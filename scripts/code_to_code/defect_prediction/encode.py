import os
import json
import argparse
import subprocess
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm


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

    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def initializer(self):
        pass

    def _encode(self, line):
        return self.sp.encode(line, out_type=str)

    def _decode(self, tokens):
        return self.sp.decode(tokens)


def process(spmfile, srcfile, index_file, outdir, split, nexample=-1, max_length=510):
    index = set()
    line_count = count_file_lines(index_file)
    line_count = line_count if nexample == -1 else min(line_count, nexample)
    with open(index_file) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            index.add(int(line))

    js_all = json.load(open(srcfile))
    data = []
    for idx, js in enumerate(js_all):
        if idx in index:
            data.append((js['func'], js['target']))

    encoder = MultiprocessingEncoder(spmfile)
    with open(os.path.join(outdir, '{}.input0'.format(split)), 'w', encoding='utf-8') as fw1, \
            open(os.path.join(outdir, '{}.label'.format(split)), 'w', encoding='utf-8') as fw2:
        for idx, ex in enumerate(tqdm(data, total=len(data))):
            if nexample != -1 and idx >= nexample:
                break
            code, label = ex[0], ex[1]
            if label not in [0, 1]:
                continue
            code_tokens = encoder._encode(code)[:max_length]
            fw1.write(' '.join(code_tokens) + '\n')
            fw2.write(str(label) + '\n')


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
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    process(
        args.model_file, args.src_file, args.index_file,
        args.output_dir, args.split, args.nexample, args.max_length
    )


if __name__ == "__main__":
    main()
