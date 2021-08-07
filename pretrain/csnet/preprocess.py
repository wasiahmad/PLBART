import re
import os
import glob
import json
import pickle
import argparse
import subprocess

from tqdm import tqdm
from data.github.preprocessing.src.code_tokenizer import tokenize_java, tokenize_python


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def prepare(args):
    src_dir = os.path.join(args.source_dir, args.lang)
    pl_writer = open(
        '{}/{}.functions.tok'.format(args.target_pl_dir, args.split), 'w', encoding='utf-8'
    )
    nl_writer = open(
        '{}/{}.docstring.tok'.format(args.target_nl_dir, args.split), 'a', encoding='utf-8'
    )
    for file in glob.glob("{}/{}_{}_*.jsonl".format(src_dir, args.lang, args.split)):
        filename, _ = os.path.splitext(os.path.basename(file))
        with open(file) as f:
            for line in tqdm(
                    f, total=count_file_lines(file), desc="{}-{}".format(args.lang, filename)
            ):
                ex = json.loads(line.strip())
                code = ' '.join(ex['code_tokens'])
                code = re.sub("[\n\r\t ]+", " ", code).strip()

                if args.lang == 'python' or args.lang == 'java':
                    _tokens = tokenize_python(ex['code']) \
                        if args.lang == 'python' else tokenize_java(ex['code'])
                    tokenized_code = ' '.join(_tokens)
                    tokenized_code = re.sub("[\n\r\t ]+", " ", tokenized_code).strip()
                    if len(tokenized_code) > 0:
                        try:
                            pl_writer.write(tokenized_code + '\n')
                        except:
                            if len(code) > 0:
                                pl_writer.write(code + '\n')
                    elif len(code) > 0:
                        pl_writer.write(code + '\n')

                elif len(code) > 0:
                    pl_writer.write(code + '\n')

                docstring = ' '.join(ex['docstring_tokens'])
                docstring = re.sub("[\n\r\t ]+", " ", docstring).strip()
                if len(docstring) > 0:
                    nl_writer.write(docstring + '\n')

    if args.split == 'train':
        num_unimodal_ex = 0
        with open("{}/{}_dedupe_definitions_v2.pkl".format(src_dir, args.lang), 'rb') as f:
            data = pickle.load(f)
            for ex in tqdm(
                    data, total=len(data), desc="unimodal-data"
            ):
                if len(ex['docstring_tokens']) == 0:
                    # unimodal data / only function
                    if 'function_tokens' in ex:
                        code = ' '.join(ex['function_tokens'])
                        code = re.sub("[\n\r\t ]+", " ", code)
                        tokenized_code = None
                        if args.lang == 'python' or args.lang == 'java':
                            _tokens = tokenize_python(ex['function']) \
                                if args.lang == 'python' else tokenize_java(ex['function'])
                            if len(_tokens) == 0:
                                continue
                            tokenized_code = ' '.join(_tokens)
                            tokenized_code = re.sub("[\n\r\t ]+", " ", tokenized_code)
                        try:
                            if tokenized_code is not None:
                                # this line can throw error `UnicodeEncodeError`
                                pl_writer.write(tokenized_code.strip() + '\n')
                            else:
                                pl_writer.write(code.strip() + '\n')
                        except:
                            pl_writer.write(code.strip() + '\n')

                        num_unimodal_ex += 1

        print('#unimodal_examples - ', num_unimodal_ex)

    pl_writer.close()
    nl_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", help='Language name',
    )
    parser.add_argument(
        "--source_dir", help='Source directory',
    )
    parser.add_argument(
        "--target_pl_dir", help="Output directory to save tokenized functions",
    )
    parser.add_argument(
        "--target_nl_dir", help="Output directory to save tokenized docstrings",
    )
    parser.add_argument(
        "--split", type=str, default='train', help='Dataset split',
    )
    args = parser.parse_args()
    prepare(args)
