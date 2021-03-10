import sentencepiece as spm

FILES = []

GITHUB_DIR = '/local/username/github_data'
STACKOVERFLOW_DIR = '/local/username/stackoverflow/desc_shards'
num_split = 4
for i in range(num_split):
    FILES.append('{}/java/train.{}.functions_class.tok'.format(GITHUB_DIR, i))
    FILES.append('{}/java/train.{}.functions_standalone.tok'.format(GITHUB_DIR, i))
    FILES.append('{}/python/train.{}.functions_class.tok'.format(GITHUB_DIR, i))
    FILES.append('{}/python/train.{}.functions_standalone.tok'.format(GITHUB_DIR, i))

FILES += [
    '{}/java/valid.functions_class.tok'.format(GITHUB_DIR),
    '{}/java/valid.functions_standalone.tok'.format(GITHUB_DIR),
    '{}/java/test.functions_class.tok'.format(GITHUB_DIR),
    '{}/java/test.functions_standalone.tok'.format(GITHUB_DIR),
    '{}/python/valid.functions_class.tok'.format(GITHUB_DIR),
    '{}/python/valid.functions_standalone.tok'.format(GITHUB_DIR),
    '{}/python/test.functions_class.tok'.format(GITHUB_DIR),
    '{}/python/test.functions_standalone.tok'.format(GITHUB_DIR),
]

FILES += [
    '{}/train.0.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/train.1.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/train.2.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/train.3.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/train.4.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/train.5.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/train.6.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/train.7.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/valid.description.txt'.format(STACKOVERFLOW_DIR),
    '{}/test.description.txt'.format(STACKOVERFLOW_DIR)
]

spm.SentencePieceTrainer.train(
    '--input={} --vocab_size=50000 --model_prefix=sentencepiece.bpe '
    '--character_coverage=1.0 --model_type=bpe'.format(','.join(FILES))
)
