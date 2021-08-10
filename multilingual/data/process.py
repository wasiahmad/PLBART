import re
import json
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


def prepare():
    for lang in ['go', 'java', 'python', 'ruby', 'javascript', 'php']:
        for split in ['train', 'valid', 'test']:
            src_writer = open(
                'processed/{}.{}-en_XX.{}'.format(split, lang, lang), 'w', encoding='utf-8'
            )
            tgt_writer = open(
                'processed/{}.{}-en_XX.en_XX'.format(split, lang), 'w', encoding='utf-8'
            )
            filename = '{}/{}.jsonl'.format(lang, split)
            with open(filename) as f:
                for line in tqdm(
                        f, total=count_file_lines(filename), desc="{}-{}".format(lang, split)
                ):
                    ex = json.loads(line.strip())
                    code = ' '.join(ex['code_tokens'])
                    code = re.sub("[\n\r\t ]+", " ", code).strip()
                    docstring = ' '.join(ex['docstring_tokens'])
                    docstring = re.sub("[\n\r\t ]+", " ", docstring).strip()
                    if len(code) == 0 or len(docstring) == 0:
                        continue

                    if lang == 'python' or lang == 'java':
                        _tokens = tokenize_python(ex['code']) \
                            if lang == 'python' else tokenize_java(ex['code'])
                        tokenized_code = ' '.join(_tokens)
                        tokenized_code = re.sub("[\n\r\t ]+", " ", tokenized_code).strip()
                        if len(tokenized_code) == 0:
                            continue
                        try:
                            # this line can throw error UnicodeEncodeError
                            src_writer.write(tokenized_code + '\n')
                        except:
                            src_writer.write(code + '\n')
                    else:
                        src_writer.write(code + '\n')

                    tgt_writer.write(docstring + '\n')

            src_writer.close()
            tgt_writer.close()


if __name__ == '__main__':
    prepare()
