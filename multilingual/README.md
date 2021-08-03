# Multilingual Multi-task Learning

We investigate multilingual multi-task learning for code summarization and generation, motivated by 
the work, [Multilingual Translation with Extensible Multilingual Pretraining and
Finetuning](https://arxiv.org/pdf/2008.00401.pdf).  

- Our investigation is based on the 
[CodeSearchNet](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text#dataset)
dataset used for code summarization task in [CodeXGlue](https://github.com/microsoft/CodeXGLUE). 
The dataset consists of six languages - java, python, javascript, php, ruby, and go.

- We fine-tune PLBART via single-task and multi-task learning. While in single-task learning, we fine-tuning PLBART on 
code summarization and code generation tasks individually. In multi-task learning, we fine-tune PLBART jointly on 
both the tasks.


### Single Language vs. Multilingual Fine-tuning

- Fine-tuning PLBART on one language, e.g., Java, Python, etc.
- Multilingual fine-tuning: fine-tuning PLBART on more than one languages.
    - On languages groups.
        - Compiled vs. Interpreted - **[Java, Ruby, go]** vs. **[PHP, Python, Javascript]**
        - Static vs. Dynamic - **[Java, Go]** vs. **[Javascript, Python, PHP, Ruby]**
        - Strongly typed vs. Weakly typed - **[Java, Go, Python, Ruby]** vs. **[PHP, Javascript]**
    - All the six languages.


### Preparation

To download and prepare the data, do the following.

```bash
cd data
bash download.sh
bash prepare.sh
cd ..
```

For multilingual multi-task fine-tuning, we need to make some modifications to PLBART pre-trained checkpoint. To do 
that, do the following.

```bash
cd plbart
python convert.py
cd ..
```


### Multilingual Multi-task Fine-tuning

All the fine-tuned checkpoints are released [**here**](https://drive.google.com/drive/folders/1j_uEjBehqqxfPTjFTOM2RyqojU3S1t0f).

#### Single-task Fine-tuning

```bash
cd single_task              
# summarization
bash summarization.sh GPU_ID LANGUAGE_GROUP LANG
# generation
bash generation.sh GPU_ID LANGUAGE_GROUP LANG
cd ..
```

#### Multi-task Fine-tuning

```bash
cd multi_task
bash run.sh GPU_ID LANGUAGE_GROUP LANG
cd ..
```

Where,

```text
GPU_ID = index of the gpu, e.g., 0, 1, 2
LANGUAGE_GROUP = [one|compiled|interpreted|static|dynamic|strong|weak|all]
LANG = [java|python|go|ruby|php|js]
```

#### Results

All the experiment results are available in the [**spreadsheet**](https://docs.google.com/spreadsheets/d/1TLw73DrvmLgK3GZ0YMGt_oX-d8t4LdH8v5OTPkb3_jY).

#### Hardware Requirements

We conduct all experiments in 1 GeForce RTX 2080 Ti GPU (11gb memory).


### References

- https://github.com/pytorch/fairseq/tree/master/examples/multilingual
