import os
import logging
import torch
import json
import sys
import copy
import gzip
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from fairseq import utils
from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.models.bart import BARTModel

from fairseq.sequence_scorer import SequenceScorer

root_dir = Path(os.path.abspath(__file__)).parents[4]
spm_dir = root_dir.joinpath('sentencepiece/sentencepiece.bpe.model')

sys.path.append(root_dir.absolute().as_posix())
try:
    from data.github.preprocessing.src.code_tokenizer import (
        tokenize_python,
        detokenize_python
    )
    from source.completion_dataset import CodeCompletionDataset
except Exception as e:
    print("Exception: ", e)

logger = logging.getLogger(__name__)


def stream_jsonl(filename):
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def read_problems(eval_file):
    return {str(task["task_id"]): task for task in stream_jsonl(eval_file)}


def encode(
        model, sentence: str
) -> torch.LongTensor:
    tokens = model.bpe.encode(sentence)
    if len(tokens.split(" ")) > model.max_positions - 1:
        tokens = " ".join(tokens.split(" ")[: model.max_positions - 1])
    bpe_sentence = tokens + " </s>"
    tokens = model.task.source_dictionary.encode_line(
        bpe_sentence, append_eos=False, add_if_not_exist=False
    )
    return tokens.long()


def build_sample(
        model,
        src_tokens: List[torch.LongTensor],
        tgt_tokens: List[torch.LongTensor],
):
    # assert torch.is_tensor(src_tokens)
    dataset = LanguagePairDataset(
        src_tokens,
        [x.numel() for x in src_tokens],
        model.task.source_dictionary,
        tgt=tgt_tokens,
        tgt_sizes=[x.numel() for x in tgt_tokens],
        tgt_dict=model.task.target_dictionary,
    )
    sample = dataset.collater(dataset)
    sample = utils.apply_to_sample(lambda tensor: tensor.to(model.device), sample)
    return sample


@torch.no_grad()
def generate(
        model,
        tokens: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        use_mean_logp: bool = False,
        **kwargs,
) -> List[List[Dict[str, torch.Tensor]]]:
    sample = model._build_sample(tokens)
    # build generator using current args as well as any kwargs
    gen_args = copy.copy(model.args)
    gen_args.beam = beam
    for k, v in kwargs.items():
        setattr(gen_args, k, v)
    generator = model.task.build_generator([model.model], gen_args)
    translations = model.task.inference_step(generator, [model.model], sample)
    # translation: List[List[Dict[str, Tensor]]]
    # dict_keys(['tokens', 'score', 'attention', 'alignment', 'positional_scores'])

    if verbose:
        src_str_with_unk = model.string(tokens)
        logger.info("S\t{}".format(src_str_with_unk))

    hypos = [v for _, v in sorted(zip(sample["id"].tolist(), translations))]
    if use_mean_logp:
        scorer = SequenceScorer(model.task.target_dictionary)
        batch_hypotheses = []
        for (src_tokens, hypotheses) in zip(tokens, hypos):
            num_hypotheses = len(hypotheses)
            tgt_tokens = [x["tokens"].cpu() for x in hypotheses]
            sample = build_sample(model, [src_tokens] * num_hypotheses, tgt_tokens)
            scored_hypos = scorer.generate([model.model], sample)
            hyp_scores = [x[0]["score"].item() for x in scored_hypos]
            batch_hypotheses.append([hypotheses[idx] for idx in np.argsort(hyp_scores)])
        return batch_hypotheses
    else:
        return hypos


def sample(
        model,
        sentences: List[str],
        beam: int = 1,
        verbose: bool = False,
        use_mean_logp: bool = False,
        **kwargs
) -> List[List[str]]:
    input = [encode(model, sentence) for sentence in sentences]
    batch_outputs = generate(model, input, beam, verbose, use_mean_logp, **kwargs)
    return [[model.decode(x["tokens"]) for x in hypos] for hypos in batch_outputs]


def process_completion(
        tokenized_prompt,
        tokenized_completion,
        suppress_buggy_completion=True,
        special_process=False,
):
    if tokenized_prompt.endswith("DEDENT"):
        tokenized_prompt = tokenized_prompt[:-6]
    function = detokenize_python(tokenized_prompt + tokenized_completion)
    if suppress_buggy_completion:
        try:
            tokenize_python(function)
        except Exception as e:
            # we filter syntactically buggy completions
            return function, None

    if special_process:
        prompt = detokenize_python(tokenized_prompt)
        completion = function.replace(prompt, '')
        if not tokenized_completion.startswith("INDENT "):
            completion = "    " + completion
    else:
        if not tokenized_completion.startswith("INDENT "):
            tokenized_completion = "INDENT " + tokenized_completion
        completion = detokenize_python(tokenized_completion)
    return function, completion


def decode(args):
    bart = BARTModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data_name_or_path,
        user_dir=root_dir.joinpath('source'),
        task=args.task,
        bpe='sentencepiece',
        sentencepiece_model=spm_dir,
    )
    logger.info(f"Model loaded from {os.path.join(args.checkpoint_dir, args.checkpoint_file)}")

    bart.eval()
    if torch.cuda.is_available():
        bart = bart.cuda().half()

    human_eval_examples = read_problems(f'{args.data_dir}/HumanEval.jsonl.gz')
    predictions = []
    batch_examples = []
    output_functions = []
    for task_id in tqdm(human_eval_examples):
        prompt = human_eval_examples[task_id]["prompt"]
        tokenized_prompt = ' '.join(tokenize_python(prompt, keep_comments=True))
        batch_examples.append((task_id, tokenized_prompt))

        if len(batch_examples) % args.batch_size == 0:
            with torch.no_grad():
                hypotheses_batch = sample(bart,
                                          [ex[1] for ex in batch_examples],
                                          beam=args.beam,
                                          sampling=args.sampling,
                                          sampling_topp=args.sampling_topp,
                                          temperature=args.temperature,
                                          lenpen=args.lenpen,
                                          max_len_b=args.max_len_b,
                                          min_len=args.min_len,
                                          no_repeat_ngram_size=args.no_repeat_ngram_size,
                                          use_mean_logp=args.use_mean_logp)
            for hypotheses, (task_id, tokenized_prompt) in zip(hypotheses_batch, batch_examples):
                assert len(hypotheses) >= args.nbest
                predictions_added = 0
                for completion in hypotheses:
                    if predictions_added >= args.nbest:
                        break
                    whole_function, completion = process_completion(
                        tokenized_prompt,
                        completion,
                        suppress_buggy_completion=True,
                    )
                    if completion is not None:
                        output_functions.append(whole_function)
                        output = {"task_id": task_id, "completion": completion}
                        predictions.append(json.dumps(output))
                        predictions_added += 1
            batch_examples = []

    if len(batch_examples) != 0:
        hypotheses_batch = sample(bart,
                                  [ex[1] for ex in batch_examples],
                                  beam=args.beam,
                                  sampling=args.sampling,
                                  sampling_topp=args.sampling_topp,
                                  temperature=args.temperature,
                                  lenpen=args.lenpen,
                                  max_len_b=args.max_len_b,
                                  min_len=args.min_len,
                                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                                  use_mean_logp=args.use_mean_logp)
        for hypotheses, (task_id, tokenized_prompt) in zip(hypotheses_batch, batch_examples):
            assert len(hypotheses) >= args.nbest
            predictions_added = 0
            for completion in hypotheses:
                if predictions_added >= args.nbest:
                    break
                whole_function, completion = process_completion(
                    tokenized_prompt,
                    completion,
                    suppress_buggy_completion=True,
                )
                if completion is not None:
                    output_functions.append(whole_function)
                    output = {"task_id": task_id, "completion": completion}
                    predictions.append(json.dumps(output))
                    predictions_added += 1

    with open(args.output_file, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(predictions))
    with open(args.output_function_file, 'w', encoding='utf-8') as fout:
        delim = '\n' + '--' * 20 + '\n'
        fout.write(f"{delim}".join(output_functions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help="Path of checkpoint directory")
    parser.add_argument('--checkpoint_file', type=str, required=True,
                        help="Path of checkpoint directory")
    parser.add_argument('--task', type=str, default="translation_without_lang_token",
                        help="Task name the model is fine-tuned on")
    parser.add_argument('--data_dir', type=str, required=True, help="Path of data directory")
    parser.add_argument('--data_name_or_path', type=str, required=True, help="Path of the binary data directory")
    parser.add_argument('--output_file', type=str, required=True, help="Path of the output file")
    parser.add_argument('--sampling', action='store_true', help="Use sampling instead of beam search")
    parser.add_argument('--beam', type=int, default=1)
    parser.add_argument('--max_len_b', type=int, default=200)
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--lenpen', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--nbest', type=int, default=10)
    parser.add_argument('--sampling_topp', type=float, default=-1.0)
    parser.add_argument('--output_function_file', type=str, required=True,
                        help="Path of the output function file")
    parser.add_argument('--use_mean_logp', action='store_true',
                        help="Use mean log probability to score decoded sequences")
    args = parser.parse_args()

    if args.beam < args.nbest:
        logger.warning("beam must be great or equal to nbest")
        args.beam = args.nbest

    decode(args)
