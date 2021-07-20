# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import sys

sys.path.append("../..")

import json
import logging
import argparse
from evaluation.bleu import compute_bleu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--expected', '-a', required=True, help="filename of the labels, in test format.")
    parser.add_argument('--predicted', '-p', required=True,
                        help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = open(args.predicted, "r").readlines()
    gts = open(args.expected, "r").readlines()

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0

    translations = []
    references = []
    for pred, gt in zip(preds, gts):
        pred = pred.strip()
        gt = json.loads(gt.strip())['code']
        pred = ' '.join([tok.strip() for tok in pred.split()])
        translations.append(pred.split())
        gt = ' '.join([tok.strip() for tok in gt.split()])
        references.append([gt.split()])
        if pred == gt:
            EM += 1

    bleu_score, _, _, _, _, _ = compute_bleu(references, translations, 4, True)
    EM = round(EM / total * 100, 2)
    bleu_score = round(100 * bleu_score, 2)
    print(f"BLEU: {bleu_score}, EM: {EM}")


if __name__ == "__main__":
    main()
