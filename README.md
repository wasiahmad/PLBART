# [PLBART](https://www.aclweb.org/anthology/2021.naacl-main.211/)

Official code release of our NAACL 2021 work, [Unified Pre-training for Program Understanding and Generation](https://www.aclweb.org/anthology/2021.naacl-main.211/). 
We present **PLBART** that is pre-trained on a large collection Java and Python functions and natural language descriptions collected from Github and StackOverflow, respectively.

We present the file structure of this repository [here](https://github.com/wasiahmad/PLBART/blob/main/FILEs.md).

### [Pre-training]()

#### Step1. Download Github data

Go to `data/github` directory and follow instructions.

#### Step2. Download StackOverflow data

Go to `data/stackoverflow` directory and follow instructions.

#### Step3. Binarize the data and pre-train

```bash
cd pretrain
bash binarize.sh
bash absolute.sh GPU_IDS
```

Note. We pre-trained PLBART on 8 `GeForce RTX 2080` (11gb) GPUs (took 11.5 days).


### [Fine-tuning on Downstream Tasks]()

We fine-tune and evaluate PLBART on three types of tasks.

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Task</th>
            <th>Language(s)</th>
            <th>Data</th>
            <th>Scripts</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Code to Text</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text" target="_blank">Code summarization</a></td>
            <td>Python, PHP, Go, Java, <br> Javascript, Ruby</td>
            <td><a href="https://drive.google.com/file/d/1m1IvGgPhDBg-SL-LajtFGTLyAJVbD0i3" target="_blank">code-to-text.zip</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_text">[LINK]</a></td>
        </tr>
        <tr>
            <td>Text to Code</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code" target="_blank">Code generation</a></td>
            <td>Java</td>
            <td><a href="https://drive.google.com/file/d/1rQjQh4Mle3yYzQbn-CRs4L1moZaAqr90" target="_blank">text-to-code.zip</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/text_to_code">[LINK]</a></td>
        </tr>
        <tr>
            <td rowspan=4>Code-to-Code</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans" target="_blank">Code translation</a></td>
            <td>Java, C#</td>
            <td rowspan=4><a href="https://drive.google.com/file/d/15jokCxFQ9BUbptMsrfj4RdH_KiNkTRP2" target="_blank">code-to-code.zip</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/translation">[LINK]</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement" target="_blank">Code refinement</a></td>
            <td>Java</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/refinement">[LINK]</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench" target="_blank">Clone detection</a></td>
            <td>Java</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/clone_detection">[LINK]</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection" target="_blank">Defect detection</a></td>
            <td>C/C++</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/defect_prediction">[LINK]</a></td>
        </tr>
    </tbody>
</table>

#### Step1. Download PLBART checkpoint

```bash
cd pretrain
bash download.sh
cd ..
```

#### Step2. Download the data

```bash
cd data/codeXglue
bash download.sh
cd ../..
```

#### Step3. Prepare the data, train and evaluate PLBART

For example, we want to fine-tune PLBART on `Text-to-Code` task. Then,

```bash
cd scripts/text_to_code
bash prepare.sh
bash run.sh GPU_IDS
cd ../..
```

Note. We fine-tuned PLBART on 1 `GeForce RTX 2080` (11gb) GPU.


### [Acknowledgement]()

PLBART uses [Fairseq](https://github.com/pytorch/fairseq), [codeXglue](https://github.com/microsoft/CodeXGLUE), and [TransCoder](https://github.com/facebookresearch/TransCoder) and thanks the authors of these works for their contribution.


### [Citation]()

```
@inproceedings{ahmad-etal-2021-unified,
    title = "Unified Pre-training for Program Understanding and Generation",
    author = "Ahmad, Wasi  and
      Chakraborty, Saikat  and
      Ray, Baishakhi  and
      Chang, Kai-Wei",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.211",
    pages = "2655--2668"
}
```

