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
            lang_iso = 'js' if lang == 'javascript' else lang
            src_writer = open(
                'processed/{}.{}-en_XX.{}'.format(split, lang_iso, lang_iso), 'w', encoding='utf-8'
            )
            tgt_writer = open(
                'processed/{}.{}-en_XX.en_XX'.format(split, lang_iso), 'w', encoding='utf-8'
            )
            filename = '{}/{}.jsonl'.format(lang, split)
            with open(filename) as f:
                for line in tqdm(
                        f, total=count_file_lines(filename), desc="{}-{}".format(lang, split)
                ):
                    ex = json.loads(line.strip())
                    try:
                        if lang == 'python' or lang == 'java':
                            code_tokens = tokenize_python(ex['code']) \
                                if lang == 'python' else tokenize_java(ex['code'])
                            if len(code_tokens) > 0:
                                raise ValueError('Empty tokenized code')
                        else:
                            code_tokens = ex['code_tokens']
                    except:
                        code_tokens = ex['code_tokens']

                    code = ' '.join(code_tokens)
                    code = re.sub("[\n\r\t ]+", " ", code)
                    docstring = ' '.join(ex['docstring_tokens'])
                    docstring = re.sub("[\n\r\t ]+", " ", docstring)
                    if len(code) > 0 and len(docstring) > 0:
                        src_writer.write(code.strip() + '\n')
                        tgt_writer.write(docstring.strip() + '\n')

            src_writer.close()
            tgt_writer.close()


if __name__ == '__main__':
    prepare()
