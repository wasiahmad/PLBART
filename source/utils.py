import re
import os
import torch
import random
import logging
from pathlib import Path
import sentencepiece as spm

logger = logging.getLogger(__name__)
root_dir = Path(os.path.abspath(__file__)).parents[1]
spm_model = root_dir.joinpath('sentencepiece/sentencepiece.bpe.model')


class SplitFunction(object):
    def __init__(self, logic, dictionary, args):
        self.logic = logic
        self.args = args
        self.dictionary = dictionary
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(spm_model.absolute().as_posix())

    def _decode(self, tokens):
        s = self.tokenizer.decode(
            [self.dictionary[t] for t in tokens]
        )
        return s  # a string

    def _encode(self, text):
        tokens = self.tokenizer.encode(text, out_type=str)
        ids = []
        for i, token in enumerate(tokens):
            idx = self.dictionary.index(token)
            ids.append(idx)
        return ids

    def logic_random(self, function):
        fn_tokens = function.split()
        split_idx = random.randint(0, len(fn_tokens) - 1)
        src_tokens, tgt_tokens = fn_tokens[:split_idx], fn_tokens[split_idx:]
        source, target = " ".join(src_tokens), " ".join(tgt_tokens)
        return source, target

    def logic_percentage(self, function):
        fn_tokens = function.split()
        assert 0.0 < self.args.percentage < 1.0
        split_idx = int(len(fn_tokens) * self.args.percentage)
        src_tokens, tgt_tokens = fn_tokens[:split_idx], fn_tokens[split_idx:]
        source, target = " ".join(src_tokens), " ".join(tgt_tokens)
        return source, target

    def logic_strategy1(self, function):
        choice = random.randint(0, 3)
        if choice == 0:
            # Source (function signature + docstring)
            # def test_plugin_inheritance(self):
            #     """ Test that an object derived from BasePlugin works properly """
            # ----------------------------------------
            # Target (function body)
            # simple_plugin = self.SimplePlugin()
            # self.assertEqual(simple_plugin.routes(), [])
            sig_ds = get_signature_and_docstrings(function)
            if len(sig_ds) == 0:
                # we are unable to apply the target split logic
                return self.logic_random(function)
            sig_ds_end_position = function.rindex(sig_ds[0]) + len(sig_ds[0])
            source = function[:sig_ds_end_position]
            target = function[sig_ds_end_position:]
        elif choice == 1:
            # Source (function signature)
            # def test_plugin_inheritance(self):
            # ----------------------------------------
            # Target (function body)
            #     """ Test that an object derived from BasePlugin works properly """
            #     simple_plugin = self.SimplePlugin()
            #     self.assertEqual(simple_plugin.routes(), [])
            sig = get_signature(function)
            if len(sig) == 0:
                # we are unable to apply the target split logic
                return self.logic_random(function)
            sig_end_position = function.rindex(sig[0]) + len(sig[0])
            source = function[:sig_end_position]
            target = function[sig_end_position:]
        elif choice == 2:
            # Source (docstring)
            # Test that an object derived from BasePlugin works properly
            # ----------------------------------------
            # Target (whole function)
            # def test_plugin_inheritance(self):
            #     simple_plugin = self.SimplePlugin()
            #     self.assertEqual(simple_plugin.routes(), [])
            sig_ds = get_signature_and_docstrings(function)
            if len(sig_ds) == 0:
                # we are unable to apply the target split logic
                return self.logic_random(function)
            sig_ds_end_position = function.rindex(sig_ds[0]) + len(sig_ds[0])
            ds = get_docstrings(function)[0]
            source = ds[18:-9].replace("\"\"\"", "").strip()
            sig = get_signature(function)[0]
            target = sig + function[sig_ds_end_position:]
        else:
            # Source (random split)
            # def test_plugin_inheritance(self):
            #     """ Test that an object derived from BasePlugin works properly """
            #     simple_plugin =
            # ----------------------------------------
            # Target (function \ Source)
            #     self.SimplePlugin()
            #     self.assertEqual(simple_plugin.routes(), [])
            return self.logic_random(function)

        return source, target

    def split(self, tokens):
        assert isinstance(tokens, torch.LongTensor)
        logic_fn = getattr(self, f"logic_{self.logic}")
        function = self._decode(tokens.tolist())
        source, target = logic_fn(function)
        source, target = self._encode(source), self._encode(target)
        source, target = torch.LongTensor(source), torch.LongTensor(target)
        return source, target


def get_signature(function):
    # e.g.,
    # def interpolate_loc ( grid , loc ) : NEW_LINE INDENT
    sig = re.findall(
        "[d][e][f][ ].*?[ ][(].*?[)][ ]"
        "[:][ ][N][E][W][_][L][I][N][E][ ][I][N][D][E][N][T]",
        function, re.DOTALL
    )
    return sig


def get_docstrings(function):
    # e.g.,
    # : NEW_LINE INDENT """ Helper which interpolates between two locs . """ NEW_LINE
    ds = re.findall(
        "[:][ ][N][E][W][_][L][I][N][E][ ][I][N][D][E][N][T][ ]['][']['].*?['][']['][ ][N][E][W][_][L][I][N][E]"
        "|"
        "[:][ ][N][E][W][_][L][I][N][E][ ][I][N][D][E][N][T][ ][\"][\"][\"].*?[\"][\"][\"][ ][N][E][W][_][L][I][N][E]",
        function, re.DOTALL
    )
    return ds


def get_signature_and_docstrings(function):
    # e.g.,
    # def interpolate_loc ( grid , loc ) : NEW_LINE INDENT """ Helper which interpolates between two locs . """ NEW_LINE
    sds = re.findall(
        "[d][e][f][ ].*?[ ][(].*?[)][ ]"
        "[:][ ][N][E][W][_][L][I][N][E][ ][I][N][D][E][N][T][ ]['][']['].*?['][']['][ ][N][E][W][_][L][I][N][E]"
        "|"
        "[d][e][f][ ].*?[ ][(].*?[)][ ]"
        "[:][ ][N][E][W][_][L][I][N][E][ ][I][N][D][E][N][T][ ][\"][\"][\"].*?[\"][\"][\"][ ][N][E][W][_][L][I][N][E]",
        function, re.DOTALL
    )
    return sds
