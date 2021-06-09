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

    def __init__(self, model_file):
        self.model_file = model_file

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
        assert isinstance(example, dict)
        code_tokens = self._encode(example['func'])
        return {'idx': example['idx'], 'code': " ".join(code_tokens)}


def preprocess(spmfile, srcfile, tgtfile, workers=20):
    dataset = []
    with open(srcfile, encoding='utf8') as f:
        for line in f:
            ex = json.loads(line)
            dataset.append(ex)

    encoder = MultiprocessingEncoder(spmfile)
    pool = Pool(workers, initializer=encoder.initializer)

    processed_dataset = {}
    with tqdm(total=len(dataset), desc='Processing') as pbar:
        for i, ex in enumerate(pool.imap(encoder.encode, dataset, 100)):
            pbar.update()
            processed_dataset[ex['idx']] = ex['code']

    with open(tgtfile, 'w', encoding='utf8') as fw:
        for item in sorted(processed_dataset.items()):
            fw.write(item[0] + '\t' + item[1] + '\n')


def postprocess(srcfile, index_file, outdir, split, nexample=-1, max_length=510):
    data = {}
    with open(srcfile, encoding='utf8') as f:
        for line in f:
            splits = line.strip().split('\t')
            assert len(splits) == 2
            data[int(splits[0])] = splits[1].split()

    line_count = count_file_lines(index_file)
    line_count = line_count if nexample == -1 else min(line_count, nexample)
    with open(os.path.join(outdir, '{}.input0'.format(split)), 'w', encoding='utf-8') as fw1, \
            open(os.path.join(outdir, '{}.label'.format(split)), 'w', encoding='utf-8') as fw2:
        with open(index_file, encoding='utf8') as f:
            for idx, line in enumerate(tqdm(f, total=line_count)):
                splits = line.strip().split('\t')
                assert len(splits) == 3
                i1_idx, i2_idx, label = int(splits[0]), int(splits[1]), int(splits[2])
                if label not in [0, 1]:
                    continue
                if nexample != -1 and idx >= nexample:
                    break

                code1_tokens = data[i1_idx]
                code2_tokens = data[i2_idx]
                total_tokens = len(code1_tokens) + len(code2_tokens) + 1  # 1 special token  to separate them
                num_tokens_to_remove = total_tokens - max_length
                for _ in range(num_tokens_to_remove):
                    if len(code1_tokens) > len(code2_tokens):
                        code1_tokens = code1_tokens[:-1]
                    else:
                        code2_tokens = code2_tokens[:-1]
                code_tokens = code1_tokens + ['</s>'] + code2_tokens
                fw1.write(' '.join(code_tokens) + '\n')
                fw2.write(str(label) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess", action='store_true',
        default=False
    )
    parser.add_argument(
        "--postprocess", action='store_true',
        default=False
    )
    parser.add_argument(
        "--model_file",
        type=str,
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
    assert not (args.preprocess and args.postprocess)
    if args.preprocess:
        preprocess(args.model_file, args.src_file, args.tgt_file, args.workers)
    if args.postprocess:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        postprocess(
            args.src_file, args.index_file, args.output_dir,
            args.split, args.nexample, args.max_length
        )


if __name__ == "__main__":
    main()
