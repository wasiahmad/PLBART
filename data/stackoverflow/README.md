### Collect and Preprocess Stack Overflow Data

We collect the StackOverflow posts (include both questions and answers, exclude code snippets) by downloading the data
dump (date: 7th September 2020) from [stackexchange](https://archive.org/download/stackexchange).

```
python preprocess.py \
    --src_dir path_2_dir_containing_so_dump;
```

Download the StackOverflow dump and save it under `root/data/stackoverflow` directory.
