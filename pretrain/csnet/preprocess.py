import re
import os
import glob
import json
import pickle
import argparse
import subprocess

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from data.github.preprocessing.src.code_tokenizer import tokenize_java, tokenize_python


def process_bimodal_instance(ex):
    code = ' '.join(ex['code_tokens'])
    code = re.sub("[\n\r\t ]+", " ", code).strip()
    docstring = ' '.join(ex['docstring_tokens'])
    docstring = re.sub("[\n\r\t ]+", " ", docstring).strip()

    tokenized_code = ''
    if args.lang == 'python' or args.lang == 'java':
        _tokens = tokenize_python(ex['code']) \
            if args.lang == 'python' else tokenize_java(ex['code'])
        tokenized_code = ' '.join(_tokens)
        tokenized_code = re.sub("[\n\r\t ]+", " ", tokenized_code).strip()

    return code, tokenized_code, docstring


def process_unimodal_instance(ex):
    code, tokenized_code = '', ''
    if len(ex['docstring_tokens']) == 0:
        # unimodal data / only function
        if 'function_tokens' in ex:
            code = ' '.join(ex['function_tokens'])
            code = re.sub("[\n\r\t ]+", " ", code).strip()

            if args.lang == 'python' or args.lang == 'java':
                _tokens = tokenize_python(ex['function']) \
                    if args.lang == 'python' else tokenize_java(ex['function'])
                tokenized_code = ' '.join(_tokens)
                tokenized_code = re.sub("[\n\r\t ]+", " ", tokenized_code).strip()

    return code, tokenized_code


def prepare(args):
    pool = Pool(min(cpu_count(), args.workers))
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
            data = [json.loads(line.strip()) for line in f]

        results = []
        with tqdm(total=len(data), desc="{}-{}".format(args.lang, filename)) as pbar:
            for i, out in enumerate(pool.imap(process_bimodal_instance, data, 1000)):
                pbar.update()
                results.append(out)

        for output in results:
            code, tokenized_code, docstring = output
            if len(docstring) > 0:
                nl_writer.write(docstring + '\n')
            if len(code) == 0:
                continue

            written = False
            if args.lang in ['java', 'python']:
                if tokenized_code:
                    try:
                        pl_writer.write(tokenized_code + '\n')
                        written = True
                    except:
                        pass

            if not written:
                pl_writer.write(code + '\n')

    if args.split == 'train':
        num_unimodal_ex = 0
        with open("{}/{}_dedupe_definitions_v2.pkl".format(src_dir, args.lang), 'rb') as f:
            data = pickle.load(f)

            results = []
            with tqdm(total=len(data), desc="unimodal-data") as pbar:
                for i, out in enumerate(pool.imap(process_unimodal_instance, data, 1000)):
                    pbar.update()
                    results.append(out)

            for output in results:
                code, tokenized_code = output
                if len(code) == 0:
                    continue

                num_unimodal_ex += 1
                written = False
                if args.lang in ['java', 'python']:
                    if tokenized_code:
                        try:
                            # write may through UnicodeEncodeError
                            pl_writer.write(tokenized_code + '\n')
                            written = True
                        except:
                            pass

                if not written:
                    pl_writer.write(code + '\n')

        # print('#unimodal_examples - ', num_unimodal_ex)

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
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    prepare(args)
