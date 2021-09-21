# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import json
import argparse

import evaluation.CodeBLEU.bleu as bleu
import evaluation.CodeBLEU.weighted_ngram_match as weighted_ngram_match
import evaluation.CodeBLEU.syntax_match as syntax_match
import evaluation.CodeBLEU.dataflow_match as dataflow_match

from pathlib import Path

root_directory = Path(__file__).parents[2]


def make_weights(reference_tokens, key_word_list):
    return {token: 1 if token in key_word_list else 0.2 \
            for token in reference_tokens}


def compute_codebleu(hypothesis, references, lang, params='0.25,0.25,0.25,0.25'):
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    kw_file = root_directory.joinpath("evaluation/CodeBLEU/keywords/{}.txt".format(lang))
    keywords = [x.strip() for x in open(kw_file, 'r', encoding='utf-8').readlines()]

    tokenized_refs_with_weights = \
        [
            [
                [
                    reference_tokens, make_weights(reference_tokens, keywords)
                ] for reference_tokens in reference
            ] for reference in tokenized_refs
        ]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    return code_bleu_score, (ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', type=str, nargs='+', required=True, help='reference files')
    parser.add_argument('--json_refs', action='store_true', help='reference files are JSON files')
    parser.add_argument('--hyp', type=str, required=True, help='hypothesis file')
    parser.add_argument('--lang', type=str, required=True,
                        choices=['java', 'javascript', 'c_sharp', 'php', 'go', 'python', 'ruby'],
                        help='programming language')
    parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
                        help='alpha, beta and gamma')

    args = parser.parse_args()

    # List(List(String))
    # -> length of the outer List is number of references per translation
    # -> length of the inner List is number of total examples
    pre_references = [
        [x.strip() for x in open(file, 'r', encoding='utf-8').readlines()]
        for file in args.refs
    ]
    # List(String)
    hypothesis = [x.strip() for x in open(args.hyp, 'r', encoding='utf-8').readlines()]

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            if args.json_refs:
                _ref = json.loads(pre_references[j][i])
                ref_for_instance.append(_ref['code'])
            else:
                ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)

    assert len(references) == len(pre_references) * len(hypothesis)

    # references is List(List(String)) where the inner List is a
    # list of reference translations for one example.
    code_bleu_score, (ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score) = \
        compute_codebleu(hypothesis, references, args.lang, args.params)
    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.
          format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
    print('CodeBLEU score: %.2f' % (code_bleu_score * 100.0))


if __name__ == '__main__':
    main()
