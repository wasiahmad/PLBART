import xml.etree.ElementTree as ET
import py7zr
import json
import os
import glob
import re
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

XML_SUFFIX = '<?xml version="1.0" encoding="utf-8"?>\n<posts>\n'
XML_PREFIX = '\n</posts>'

posts = {
    'Id': 'INTEGER',
    'PostTypeId': 'INTEGER',  # 1: Question, 2: Answer
    'ParentId': 'INTEGER',  # (only present if PostTypeId is 2)
    'AcceptedAnswerId': 'INTEGER',  # (only present if PostTypeId is 1)
    'CreationDate': 'DATETIME',
    'Score': 'INTEGER',
    'ViewCount': 'INTEGER',
    'Body': 'TEXT',
    'OwnerUserId': 'INTEGER',  # (present only if user has not been deleted)
    'OwnerDisplayName': 'TEXT',
    'LastEditorUserId': 'INTEGER',
    'LastEditorDisplayName': 'TEXT',  # ="Rich B"
    'LastEditDate': 'DATETIME',  # ="2009-03-05T22:28:34.823"
    'LastActivityDate': 'DATETIME',  # ="2009-03-11T12:51:01.480"
    'CommunityOwnedDate': 'DATETIME',  # (present only if post is community wikied)
    'Title': 'TEXT',
    'Tags': 'TEXT',
    'AnswerCount': 'INTEGER',
    'CommentCount': 'INTEGER',
    'FavoriteCount': 'INTEGER',
    'ClosedDate': 'DATETIME'
}


def extract_7zip(source_dir):
    with py7zr.SevenZipFile(os.path.join(source_dir, 'stackoverflow.com-Posts.7z'), mode='r') as z:
        z.extractall()


def process_xml_line(line):
    line = line.strip()
    line = XML_SUFFIX + line + XML_PREFIX

    try:
        root = ET.fromstring(line.strip())
    except:
        return None

    if root is None:
        return None
    assert len(root) == 1

    element = root[0]
    data = {}
    for key in posts.keys():
        data[key] = element.get(key)

    return data


def txt_to_json(path, outpath, filesize=1800000, max_split=8):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    file_index = 0
    pool = Pool(cpu_count())
    linecount = 0
    fw = open('{}/split-{:03d}.json'.format(outpath, file_index), 'w', encoding='utf-8')
    for file in glob.glob("{}/*.txt".format(path)):
        lines = [line for line in open(file, 'r')]
        print(file)
        with tqdm(total=len(lines), desc='Processing') as pbar:
            for data in pool.imap(process_xml_line, lines, 1000):
                pbar.update()
                if data is not None:
                    linecount += 1
                    fw.write(json.dumps(data) + '\n')
                    if linecount % filesize == 0 and file_index < max_split - 1:
                        file_index += 1
                        if not fw.closed:
                            fw.close()
                        fw = open('{}/split-{:03d}.json'.format(outpath, file_index), 'w', encoding='utf-8')

    print(linecount)
    if not fw.closed:
        fw.close()


def process_chunk(ex):
    description = ''
    if ex['Title'] is not None:
        description += ex['Title'].strip()
    soup = BeautifulSoup(ex['Body'].strip(), features="lxml")
    if soup.find('pre'):
        soup.pre.decompose()
    description += ' ' + soup.get_text()
    return description


def parse_nl_data(path, outpath):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    pool = Pool(cpu_count())
    total_files = sum(1 for _ in glob.glob("{}/*.json".format(path)))
    for part in range(total_files):
        with open('{}/split-{:03d}.json'.format(path, part), 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        results = []
        with tqdm(total=len(data), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(process_chunk, data, 1000)):
                pbar.update()
                tokens = ex.split()
                if len(tokens) > 10:
                    results.append(' '.join(tokens))

        if part == total_files - 1:
            with open('{}/test.description.txt'.format(outpath), 'w', encoding='utf-8') as fw:
                fw.write('\n'.join(results[:10000]))
            with open('{}/valid.description.txt'.format(outpath), 'w', encoding='utf-8') as fw:
                fw.write('\n'.join(results[10000:20000]))
            with open('{}/train.{}.description.txt'.format(outpath, part), 'w', encoding='utf-8') as fw:
                fw.write('\n'.join(results[20000:]))
        else:
            with open('{}/train.{}.description.txt'.format(outpath, part), 'w', encoding='utf-8') as fw:
                fw.write('\n'.join(results))


def split_xml_java_python(inpath, outpath):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    file_index = 0
    linecount = 0
    selected_ids = set()
    with open(inpath, "r") as f:
        next(f)
        fw = open('{}/questions_{}.txt'.format(outpath, file_index), 'w', encoding='utf-8')
        for line in f:
            line = line.strip()
            matches = re.findall(r'<row Id=\"(.+?)\"', line)
            assert len(matches) <= 1
            if len(matches) == 0:
                continue
            row_id = int(matches[0])

            matches = re.findall(r'Tags=\"(.+?)\"', line)
            assert len(matches) <= 1
            if len(matches) == 0:
                continue
            matches = matches[0]
            if not ('java' in matches or 'python' in matches):
                continue

            selected_ids.add(row_id)
            fw.write(line + '\n')
            linecount += 1
            if linecount % 1000000 == 0:
                file_index += 1
                if not fw.closed:
                    fw.close()
                fw = open('{}/questions_{}.txt'.format(outpath, file_index), 'w', encoding='utf-8')

    print(linecount)
    if not fw.closed:
        fw.close()

    linecount = 0
    file_index = 0
    with open(inpath, "r") as f:
        next(f)
        fw = open('{}/answers_{}.txt'.format(outpath, file_index), 'w', encoding='utf-8')
        for line in f:
            line = line.strip()
            matches = re.findall(r'ParentId=\"(.+?)\"', line)
            if len(matches) != 1:
                continue
            parent_id = int(matches[0])

            if parent_id in selected_ids:
                fw.write(line + '\n')
                linecount += 1
                if linecount % 1000000 == 0:
                    file_index += 1
                    if not fw.closed:
                        fw.close()
                    fw = open('{}/answers_{}.txt'.format(outpath, file_index), 'w', encoding='utf-8')

    print(linecount)
    if not fw.closed:
        fw.close()


def split_xml(inpath, outpath):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    file_index = 0
    linecount = 0
    with open(inpath, "r") as f:
        next(f)
        fw = open('{}/posts_{}.txt'.format(outpath, file_index), 'w', encoding='utf-8')
        for line in f:
            line = line.strip()
            fw.write(line + '\n')
            linecount += 1
            if linecount % 5000000 == 0:
                file_index += 1
                if not fw.closed:
                    fw.close()
                fw = open('{}/posts_{}.txt'.format(outpath, file_index), 'w', encoding='utf-8')

    print(linecount)
    if not fw.closed:
        fw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help='Source directory')
    args = parser.parse_args()

    # extract the 7zip file into xml (file size will be ~80GB)
    extract_7zip(args.src_dir)
    # to tackle a large file (of 80GB), we split and dump them into xml shards
    split_xml(
        '{}/Posts.xml'.format(args.src_dir), '{}/xml_shards'.format(args.src_dir)
    )
    # convert the xml-style lines into dictionary object
    # ~50M posts are saved in 8 files, each with 6.25M
    txt_to_json(
        '{}/xml_shards'.format(args.src_dir),
        '{}/json_shards'.format(args.src_dir),
        filesize=6250000
    )
    # prepare the NL examples
    parse_nl_data(
        '{}/json_shards'.format(args.src_dir), '{}/desc_shards'.format(args.src_dir)
    )
