
We tokenize all the data with a sentencepiece model [(Kudo and Richardson, 2018)](https://www.aclweb.org/anthology/D18-2012/) 
learned on 1/5' th of the pre-training  data. We train sentencepiece to learn 50,000 subword tokens.

```
python train.py \
    --git_data_dir path_2_github_data \
    --so_data_dir path_2_stackoverflow_data;
```

In our directory structure, the above command should be:

```
python train.py \
    --git_data_dir ../data/github/ \
    --so_data_dir ../data/stackoverflow/desc_shards;
```
