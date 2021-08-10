# Pre-training PLBART using CodeSearchNet

In our work, we pre-trained PLBART on a large collection of source code and natural language description from Github 
and StackOverflow. On the other hand, other pre-trained models, such as [CodeBERT](https://arxiv.org/abs/2002.08155), 
[GraphCodeBERT](https://arxiv.org/abs/2009.08366) are pre-trained on the [CodeSearchNet](https://github.com/github/CodeSearchNet) 
dataset. Therefore, we investigate PLBART's performance if pre-trained on the CodeSearchNet dataset.

To pre-train PLBART on CodeSearchNet, do the following.

```bash
bash setup.sh
bash binarize.sh
bash pretrain.sh
```

#### Pre-training Data Statistics

Number of docstring used is 1,880,853 and number of functions used are detailed below.

|               | Num Examples | 
| ------------- | ------------ |
| Java          | 1,524,722    | 
| Python        | 1,069,208    |
| Javascript    | 1,841,822    |
| PHP           | 921,770      | 
| Go            | 696,935      |
| Ruby          | 159,342      |
| Total         | 6,213,799    |

**[Note]** 

- We pre-trained PLBART on CodeSearchNet using 8 `GeForce RTX 2080` (11gb) GPUs (took ~11.5 days).
- We have published the checkpoints [here]().

### Experiments

We fine-tune PLBART-CSNet on the code summarization and generation tasks. The scripts are provided in the 
`root_directory/scripts/plbart_csnet` directory. We compare PLBART-CSNet to PLBART and the experiment results are as follows. 

#### Code to Text Generation

Dataset: [CodeSearchNet](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text)

|               | Ruby  | Javascript | Go    | Python | Java  | PHP   | Overall |
| ------------- | :---: | :--------: | :---: | :----: | :---: | :---: | :-----: |
| CodeBERT      | 12.16 |  14.90     | 18.07 | 19.06  | 17.65 | 25.16 | 17.83   |
| PLBART        | 14.11 |  15.56     | 18.91 | 19.30  | 18.45 | 23.58 | 18.32   |
| PLBART-CSNet  | xx.xx |  xx.xx     | xx.xx | xx.xx  | xx.xx | xx.xx | xx.xx   |

#### Text to Code Generation

Dataset: [Concode](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code)

|                   | EM    | BLEU  | COdeBLEU | 
| -------------     | :---: | :---: | :------: |
| GPT-2             | 17.35 | 25.37 | 29.69   |
| CodeGPT-2         | 18.25 | 28.69 | 32.71   |
| CodeGPT-adapted   | 20.10 | 32.79 | 35.98   |
| PLBART            | 18.75 | 36.69 | 38.52   |
| PLBART-CSNet      | xx.xx | xx.xx | xx.xx   |

