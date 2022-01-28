# Pre-training PLBART using CodeSearchNet

In our work, we pre-trained PLBART on a large collection of source code and natural language description from Github and
StackOverflow. On the other hand, other pre-trained models, such as [CodeBERT](https://arxiv.org/abs/2002.08155),
[GraphCodeBERT](https://arxiv.org/abs/2009.08366) are pre-trained on
the [CodeSearchNet](https://github.com/github/CodeSearchNet)
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
- We have published the checkpoint [here](https://drive.google.com/file/d/1Jmmow7g4JFw-xgJxL8jYhuWR2tb3uxQr/view).

### Experiments

- We fine-tuned PLBART-CSNet on all the downstream tasks PLBART evaluated on.
- The scripts are provided in the `root_directory/scripts/plbart_csnet` directory.
- We compare PLBART-CSNet to PLBART and the experiment results are as follows.

#### Code to Text Generation

Dataset: [CodeSearchNet](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text)

|               | Ruby  | Javascript | Go    | Python | Java  | PHP   | Overall |
| ------------- | :---: | :--------: | :---: | :----: | :---: | :---: | :-----: |
| CodeBERT      | 12.16 |  14.90     | 18.07 | 19.06  | 17.65 | 25.16 | 17.83   |
| PLBART        | 14.11 |  15.56     | 18.91 | 19.30  | 18.45 | 23.58 | 18.32   |
| PLBART-CSNet  | 14.48 |  16.00     | 17.61 | 20.07  | 19.81 | 24.48 | 18.74   |

#### Text to Code Generation

Dataset: [Concode](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code)

|                   | EM    | BLEU  | CodeBLEU | 
| -------------     | :---: | :---: | :------: |
| GPT-2             | 17.35 | 25.37 | 29.69   |
| CodeGPT-2         | 18.25 | 28.69 | 32.71   |
| CodeGPT-adapted   | 20.10 | 32.79 | 35.98   |
| PLBART            | 18.75 | 36.69 | 38.52   |
| PLBART-CSNet      | 18.60 | 36.79 | 38.81   |

#### Code to Code Generation

Task: [Translation](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans)

<table>
    <thead>
        <tr>
            <th rowspan=2 align ="left">Methods</th>
            <th colspan=3 align ="center">Java to C#</th>
            <th colspan=3 align ="center">C# to Java</th>
        </tr>
        <tr>
            <th align ="center">BLEU</th>
            <th align ="center">EM</th>
            <th align ="center">CodeBLEU</th>
            <th align ="center">BLEU</th>
            <th align ="center">EM</th>
            <th align ="center">CodeBLEU</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CodeBERT</td>
            <td align ="center">79.9</td>
            <td align ="center">59.0</td>
            <td align ="center">85.1</td>
            <td align ="center">72.1</td>
            <td align ="center">58.8</td>
            <td align ="center">79.4</td>
        </tr>
        <tr>
            <td>GraphCodeBERT</td>
            <td align ="center">80.6</td>
            <td align ="center">59.4</td>
            <td align ="center">-</td>
            <td align ="center">72.6</td>
            <td align ="center">58.8</td>
            <td align ="center">-</td>
        </tr>
        <tr>
            <td>PLBART</td>
            <td align ="center">83.0</td>
            <td align ="center">64.6</td>
            <td align ="center">87.9</td>
            <td align ="center">78.4</td>
            <td align ="center">65.0</td>
            <td align ="center">85.3</td>
        </tr>
        <tr>
            <td>PLBART-CSNet</td>
            <td align ="center">81.6</td>
            <td align ="center">61.6</td>
            <td align ="center">86.8</td>
            <td align ="center">78.0</td>
            <td align ="center">63.5</td>
            <td align ="center">84.9</td>
        </tr>
    </tbody>
</table> 

Task: [Defect Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection),
[Clone Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench)

|                   | Vulnerability <br/> Detection | Clone <br/> Detection | 
| -------------     | :---------------------: | :-------------: |
| CodeBERT          | 62.08                   | 96.5            |
| GraphCodeBERT     | -                       | 97.1            |
| PLBART            | 63.18                   | 97.2            |
| PLBART-CSNet      | 59.44                   | 97.4            |

Task: [Code Refinement](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement)

<table>
    <thead>
        <tr>
            <th rowspan=2 align ="left">Methods</th>
            <th colspan=2 align ="center">Small</th>
            <th colspan=2 align ="center">Medium</th>
        </tr>
        <tr>
            <th align ="center">EM</th>
            <th align ="center">BLEU</th>
            <th align ="center">EM</th>
            <th align ="center">BLEU</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CodeBERT</td>
            <td align ="center">16.40</td>
            <td align ="center">77.42</td>
            <td align ="center">5.16</td>
            <td align ="center">91.07</td>
        </tr>
        <tr>
            <td>GraphCodeBERT</td>
            <td align ="center">17.30</td>
            <td align ="center">80.58</td>
            <td align ="center">9.10</td>
            <td align ="center">72.64</td>
        </tr>
        <tr>
            <td>PLBART</td>
            <td align ="center">19.21</td>
            <td align ="center">77.02</td>
            <td align ="center">8.98</td>
            <td align ="center">88.50</td>
        </tr>
        <tr>
            <td>PLBART-CSNet</td>
            <td align ="center">19.13</td>
            <td align ="center">76.95</td>
            <td align ="center">11.60</td>
            <td align ="center">88.08</td>
        </tr>
    </tbody>
</table> 
