import os
import sys
import json
import torch
import logging
import numpy as np
from pathlib import Path
from argparse import Namespace
from fairseq.tasks import register_task, FairseqTask
from fairseq.data import (
    data_utils,
    encoders,
    Dictionary,
    indexed_dataset,
    AppendTokenDataset,
    TruncateDataset,
    StripTokenDataset,
    LanguagePairDataset,
)
from fairseq import metrics, options, models, utils, checkpoint_utils
from completion_dataset import CodeCompletionDataset
from utils import SplitFunction

root_dir = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(root_dir.absolute().as_posix())
try:
    from data.github.preprocessing.src.code_tokenizer import (
        tokenize_python,
        detokenize_python
    )
except Exception as e:
    print("Exception: ", e)

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)


@register_task('code_completion')
class CodeCompletionTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                                will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--show-samples-interval', type=int, default=10,
                            help='interval for showing backtranslation samples')
        #
        parser.add_argument('--split-logic', type=str, default="random",
                            help='how to split the code sequences to form pairs')
        parser.add_argument('--split-fn-args', type=str, metavar='JSON',
                            help='args for splitting sources to form pairs, if needed')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        self.langs = args.langs.split(",")
        for l in self.langs:
            self.dictionary.add_symbol("[{}]".format(l))
        self.dictionary.add_symbol("<mask>")

        self.SHOW_SAMPLES_INTERVAL = args.show_samples_interval
        self._show_samples_ctr = self.SHOW_SAMPLES_INTERVAL
        split_fn_args = json.loads(getattr(args, 'split_fn_args', '{}') or '{}')
        self.split_fn = SplitFunction(
            args.split_logic,
            self.dictionary,
            Namespace(**split_fn_args)
        ).split

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task.
        """
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = cls.load_dictionary(os.path.join(paths[0], "dict.txt"))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        dataset = CodeCompletionDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            split_fn=self.split_fn,
            shuffle=(split != 'test'),
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            append_eos_to_source=True,
            append_eos_to_target=True,
        )

        self.datasets[split] = dataset
        logger.info(
            "Split: {0}, Loaded {1} samples of CodeCompletionDataset".format(
                split,
                len(self.datasets[split]),
            )
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.dictionary,
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))
            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def display_sample(self, smp, detokenize=False):
        def decode(tensor):
            tokens = [self.dictionary[t] for t in tensor.tolist()]
            sentence = " ".join(tokens)
            sentence = sentence.replace("\u2581\u2581", "SPACETOKEN")
            sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
            sentence = sentence.replace("SPACETOKEN", " ")
            return sentence

        src_tokens = smp["net_input"]["src_tokens"][0]
        prev_output_tokens = smp["net_input"]["prev_output_tokens"][0]
        tgt_tokens = smp["target"][0]
        src_tokens = src_tokens[src_tokens != self.dictionary.pad()]
        prev_output_tokens = prev_output_tokens[prev_output_tokens != self.dictionary.pad()]
        tgt_tokens = tgt_tokens[tgt_tokens != self.dictionary.pad()]

        src_str = decode(src_tokens)
        prev_output_str = decode(prev_output_tokens)
        tgt_str = decode(tgt_tokens)
        if detokenize:
            logger.info(
                f"\n[source]\n{detokenize_python(src_str)}"
                f"\n[prev_output]\n{detokenize_python(prev_output_str)}"
                f"\n[target]\n{detokenize_python(tgt_str)}"
            )
        else:
            logger.info(
                f"\n[source]\n{src_str}"
                f"\n[prev_output]\n{prev_output_str}"
                f"\n[target]\n{tgt_str}"
            )

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        # self.display_sample(sample)
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.dictionary.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        sources, hyps, refs = [], [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(
                decode(
                    utils.strip_pad(sample['target'][i], self.dictionary.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
            src_tokens = utils.strip_pad(sample["net_input"]["src_tokens"][i], self.dictionary.pad())
            sources.append(
                decode(src_tokens[:-1], escape_unk=False)
            )

        if self.args.eval_tokenized_bleu:
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            bleu = sacrebleu.corpus_bleu(hyps, [refs])

        if self.args.eval_bleu_print_samples:
            self._show_samples_ctr += 1
            if self._show_samples_ctr >= self.SHOW_SAMPLES_INTERVAL:
                self._show_samples_ctr = 0
                logger.info(
                    f"\n[source]\n {detokenize_python(sources[0])}"
                    f"\n[hypothesis]\n {detokenize_python(hyps[0])}"
                    f"\n[reference]\n {detokenize_python(refs[0])}"
                )

        return bleu

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
