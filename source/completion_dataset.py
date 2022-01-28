import logging
import torch
import numpy as np
from fairseq.data import data_utils, FairseqDataset

logger = logging.getLogger(__name__)


def collate(
        samples,
        pad_idx,
        eos_idx,
        input_feeding=True,
        pad_to_length=None,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge(
        'source',
        left_pad=False,
        pad_to_length=pad_to_length['source']
        if pad_to_length is not None
        else None
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge(
            'target',
            left_pad=False,
            pad_to_length=pad_to_length['target']
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens', left_pad=False)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=False,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length['target']
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    return batch


class CodeCompletionDataset(FairseqDataset):
    def __init__(
            self,
            dataset,
            sizes,
            dictionary,
            split_fn=None,
            shuffle=True,
            input_feeding=True,
            eos=None,
            append_eos_to_source=False,
            append_eos_to_target=False,
            max_source_positions=None,
            max_target_positions=None,
    ):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.dictionary = dictionary
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.split_fn = split_fn
        self.eos = (eos if eos is not None else dictionary.eos())
        self.append_eos_to_source = append_eos_to_source
        self.append_eos_to_target = append_eos_to_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

    def __getitem__(self, index):
        assert self.dataset[index][-1] == self.eos
        tokens = self.dataset[index][:-1]

        source, target = self.split_fn(tokens) \
            if self.split_fn is not None \
            else (tokens, None)

        assert isinstance(source, torch.LongTensor) and source.dim() == 1
        if source.size(0) > self.max_source_positions - 1:
            # truncate from left
            start = source.size(0) - self.max_source_positions - 1
            source = source[start:start + self.max_source_positions - 1]
        if target is not None:
            assert isinstance(target, torch.LongTensor) and target.dim() == 1
            if target.size(0) > self.max_target_positions - 1:
                # truncate from right
                target = target[:self.max_target_positions - 1]

        if self.append_eos_to_source:
            source = torch.cat([source, torch.LongTensor([self.eos])])

        if self.append_eos_to_target and target is not None:
            target = torch.cat([target, torch.LongTensor([self.eos])])

        return {
            'id': index,
            'source': source,
            'target': target,
        }

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(
            samples,
            pad_idx=self.dictionary.pad(),
            eos_idx=self.eos,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.src, 'supports_prefetch')
                and self.src.supports_prefetch
                and hasattr(self.tgt, 'supports_prefetch')
                and self.tgt.supports_prefetch
        )
