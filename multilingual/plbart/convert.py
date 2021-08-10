import torch
import numpy as np


def convert():
    filename = '../../pretrain/checkpoint_11_100000.pt'
    model = torch.load(filename)
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

    # Last token is mask token, let's exclude that from being considered.
    # encoder_emb_weight = states['encoder.embed_tokens.weight'][:-1, :]

    additional_languages = 4
    extra_weight_tensors = torch.Tensor(np.random.uniform(0, 1, size=(additional_languages, 768)))

    # final_embedding = torch.cat((encoder_emb_weight, extra_weight_tensors), 0)
    # print(final_embedding.shape)  # torch.Size([50008, 768])

    def append_embedding(key):
        extra_weight_tensors = torch.Tensor(np.random.uniform(0, 1, size=(4, 768)))
        dest = torch.cat((model['model'][key][:-1, :], extra_weight_tensors), 0)
        model['model'][key] = dest
        # print(dest.shape)  # torch.Size([50008, 768])

    append_embedding('encoder.embed_tokens.weight')
    append_embedding('decoder.embed_tokens.weight')
    append_embedding('decoder.output_projection.weight')

    # states = model['model']
    # print(states['encoder.embed_tokens.weight'].shape)  # torch.Size([50008, 768])
    # print(states['decoder.embed_tokens.weight'].shape)  # torch.Size([50008, 768])
    # print(states['decoder.output_projection.weight'].shape)  # torch.Size([50008, 768])

    torch.save(model, 'multilingual_plbart.pt')


def sanity_check():
    #########################################################
    # we may do a sanity check

    from fairseq.models.bart import BARTModel

    bart = BARTModel.from_pretrained(
        '.',
        checkpoint_file='multilingual_plbart.pt',
        data_name_or_path='../data/processed/binary',
        user_dir='../../source',
        task="translation_multi_simple_epoch_extended",
        decoder_langtok=True,
        lang_pairs='java-en_XX',
        lang_dict='lang_dict.txt'
    )

    print(len(bart.task.source_dictionary))  # 50008
    print(bart.task.source_dictionary[0])  # <s>
    print(bart.task.source_dictionary[1])  # <pad>
    print(bart.task.source_dictionary[2])  # </s>
    print(bart.task.source_dictionary[3])  # <unk>
    print(bart.task.source_dictionary[50001])  # __java__
    print(bart.task.source_dictionary[50002])  # __python__
    print(bart.task.source_dictionary[50003])  # __en_XX__
    print(bart.task.source_dictionary[50004])  # __js__
    print(bart.task.source_dictionary[50005])  # __php__
    print(bart.task.source_dictionary[50006])  # __ruby__
    print(bart.task.source_dictionary[50007])  # __go__


if __name__ == '__main__':
    convert()
    sanity_check()
