import os
import torch
import argparse
import numpy as np


def convert(input_ckpt, out_dir, outfile):
    # input_ckpt = '../../pretrain/plbart_base.pt'
    model = torch.load(input_ckpt)
    # model keys - dict_keys(['args', 'model', 'optimizer_history', 'extra_state', 'last_optimizer_state'])

    # states = model['model']
    # print(states['encoder.embed_tokens.weight'].shape)  # torch.Size([50005, 768])
    # print(states['decoder.embed_tokens.weight'].shape)  # torch.Size([50005, 768])
    # print(states['decoder.output_projection.weight'].shape)  # torch.Size([50005, 768])

    # PLBART's dictionary has 49,997 tokens
    # +4 special tokens (<s>, <pad>, </s>, <unk>)
    # +3 language id tokens (<java>, <python>, <en_XX>)
    # +1 <mask> token
    # total = 50,005

    additional_languages = 4  # __js__; __php__; __ruby__; __go__

    def append_embedding(key):
        extra_weight_tensors = torch.Tensor(
            np.random.uniform(0, 1, size=(additional_languages, 768))
        )
        # Last token is mask token, let's exclude that from being considered.
        # model['model'][key][:-1, :]
        dest = torch.cat(
            (model['model'][key][:-1, :], extra_weight_tensors), 0
        )
        model['model'][key] = dest
        # print(dest.shape)  # torch.Size([50008, 768])

    append_embedding('encoder.embed_tokens.weight')
    append_embedding('decoder.embed_tokens.weight')
    append_embedding('decoder.output_projection.weight')

    # states = model['model']
    # print(states['encoder.embed_tokens.weight'].shape)  # torch.Size([50008, 768])
    # print(states['decoder.embed_tokens.weight'].shape)  # torch.Size([50008, 768])
    # print(states['decoder.output_projection.weight'].shape)  # torch.Size([50008, 768])

    torch.save(model, os.path.join(out_dir, outfile))


def sanity_check(model_name_or_path, checkpoint_file):
    from fairseq.models.bart import BARTModel

    bart = BARTModel.from_pretrained(
        model_name_or_path,
        checkpoint_file=checkpoint_file,
        data_name_or_path='../data/processed/binary',
        user_dir='../../source',
        task="translation_multi_simple_epoch_extended",
        decoder_langtok=True,
        lang_pairs='java-en_XX',
        lang_dict='lang_dict.txt'
    )

    assert len(bart.task.source_dictionary) == 50008
    assert bart.task.source_dictionary[0] == '<s>'
    assert bart.task.source_dictionary[1] == '<pad>'
    assert bart.task.source_dictionary[2] == '</s>'
    assert bart.task.source_dictionary[3] == '<unk>'
    assert bart.task.source_dictionary[50001] == '__java__'
    assert bart.task.source_dictionary[50002] == '__python__'
    assert bart.task.source_dictionary[50003] == '__en_XX__'
    assert bart.task.source_dictionary[50004] == '__javascript__'
    assert bart.task.source_dictionary[50005] == '__php__'
    assert bart.task.source_dictionary[50006] == '__ruby__'
    assert bart.task.source_dictionary[50007] == '__go__'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Modify embeddings in PLBART's pre-trained checkpoint."
    )
    parser.add_argument(
        '--input_ckpt',
        default='../../pretrain/plbart_base.pt',
        help="input checkpoint file."
    )
    parser.add_argument(
        '--output_dir',
        default='.',
        help="output directory to store the checkpoint."
    )
    parser.add_argument(
        '--output_ckpt',
        default='plbart_base_multilingual.pt',
        help="output checkpoint file."
    )
    args = parser.parse_args()
    convert(args.input_ckpt, args.output_dir, args.output_ckpt)
    sanity_check(args.output_dir, args.output_ckpt)
