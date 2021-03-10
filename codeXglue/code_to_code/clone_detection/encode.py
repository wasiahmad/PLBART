import os
import json
import argparse
import subprocess
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm

HOME_DIR = '/home/username/workspace/projects/CodeBART/data/codeXglue/code-to-code/clone_detection'
SPM_FILE = '/local/username/codebart/sentencepiece.bpe.model'


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


def preprocess(spmfile, srcfile, tgtfile):
    encoder = MultiprocessingEncoder(spmfile)
    dataset = {}
    fn_lengths = []
    with open(srcfile, encoding='utf8') as f:
        for line in tqdm(f, total=count_file_lines(srcfile)):
            ex = json.loads(line)
            code = encoder._encode(ex['func'])
            dataset[ex['idx']] = " ".join(code)
            fn_lengths.append(len(code))

    print('Max/Avg. function length  - {} / {}'.format(max(fn_lengths), 1.0 * sum(fn_lengths) / len(fn_lengths)))
    with open(tgtfile, 'w', encoding='utf8') as fw:
        for item in sorted(dataset.items()):
            fw.write(item[0] + '\t' + item[1] + '\n')


def postprocess(srcfile, tgtfile, outdir, split, nexample=-1, max_length=510):
    data = {}
    with open(srcfile, encoding='utf8') as f:
        for line in f:
            splits = line.strip().split('\t')
            assert len(splits) == 2
            data[int(splits[0])] = splits[1].split()

    line_count = count_file_lines(tgtfile)
    line_count = line_count if nexample == -1 else min(line_count, nexample)
    with open(os.path.join(outdir, '{}.input0'.format(split)), 'w', encoding='utf-8') as fw1, \
            open(os.path.join(outdir, '{}.label'.format(split)), 'w', encoding='utf-8') as fw2:
        with open(tgtfile, encoding='utf8') as f:
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
        default=SPM_FILE,
        help='path to *.model file',
    )
    parser.add_argument(
        "--src_file",
        type=str,
        default=os.path.join(HOME_DIR, 'data.jsonl'),
        help="input files (.jsonl) to filter/encode",
    )
    parser.add_argument(
        "--tgt_file",
        type=str,
        default=os.path.join(HOME_DIR, 'data-processed.txt'),
        help="output file (.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=HOME_DIR + '/processed',
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
    assert not (args.preprocess and args.postprocess)
    if args.preprocess:
        preprocess(args.model_file, args.src_file, args.tgt_file)
    if args.postprocess:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        postprocess(
            args.src_file, args.tgt_file, args.output_dir,
            args.split, args.nexample, args.max_length
        )


if __name__ == "__main__":
    main()
