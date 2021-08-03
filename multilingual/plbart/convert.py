import torch
import numpy as np

filename = '../../pretrain/checkpoint_11_100000.pt'
model = torch.load(filename)
# model keys - dict_keys(['args', 'model', 'optimizer_history', 'extra_state', 'last_optimizer_state'])

# states = model['model']
# print(states['encoder.embed_tokens.weight'].shape)  # torch.Size([50005, 768])
# print(states['decoder.embed_tokens.weight'].shape)  # torch.Size([50005, 768])
# print(states['decoder.output_projection.weight'].shape)  # torch.Size([50005, 768])

# PLBART's dictionary has 49,997 tokens
# +4 special tokens (<s>, </s>, <unk>, <pad>)
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
