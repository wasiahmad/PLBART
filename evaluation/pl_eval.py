# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import argparse

from evaluation.bleu import compute_bleu
from evaluation.CodeBLEU.calc_code_bleu import compute_codebleu
from data.github.preprocessing.src.code_tokenizer import (
    detokenize_python,
    detokenize_java
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--references', required=True, help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', required=True,
                        help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument('--lang', type=str, required=True, help='name of the programming language',
                        choices=['java', 'javascript', 'php', 'go', 'python', 'ruby'])
    parser.add_argument('--detokenize', action='store_true',
                        help="detokenize both predictions and reference code.")
    args = parser.parse_args()

    refs = [x.strip() for x in open(args.references, 'r', encoding='utf-8').readlines()]
    preds = [x.strip() for x in open(args.predictions, 'r', encoding='utf-8').readlines()]

    assert len(refs) == len(refs)

    total = len(refs)
    EM = 0.0

    translations = []
    references = []
    split_translations = []
    split_references = []
    for pred, ref in zip(preds, refs):
        if args.detokenize:
            if args.lang == 'java':
                pred = detokenize_java(pred)
                ref = detokenize_java(ref)
            elif args.lang == 'python':
                pred = detokenize_python(pred)
                ref = detokenize_python(ref)

        if pred == ref:
            EM += 1

        translations.append(pred)
        references.append([ref])
        split_translations.append(pred.split())
        split_references.append([ref.split()])

    bleu_score, _, _, _, _, _ = compute_bleu(split_references, split_translations, 4, True)
    EM = round(EM / total * 100, 2)
    bleu_score = round(100 * bleu_score, 2)
    print(f"BLEU: {bleu_score}, EM: {EM}")
    code_bleu_score, _ = compute_codebleu(translations, references, args.lang)
    print('CodeBLEU score: %.2f' % (code_bleu_score * 100.0))


if __name__ == "__main__":
    main()
