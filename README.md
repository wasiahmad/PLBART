# PLBART

Official code release of our NAACL 2021 work, [Unified Pre-training for Program Understanding and Generation](https://www.aclweb.org/anthology/2021.naacl-main.211/). 
We present **PLBART** that is pre-trained on a large collection Java and Python functions and natural language descriptions collected from Github and StackOverflow, respectively.

We present the file structure of this repository [here](https://github.com/wasiahmad/PLBART/blob/main/FILEs.md).

### What's New:

- August 2021 - [Multilingual multi-task learning using PLBART](https://github.com/wasiahmad/PLBART/blob/main/multilingual/README.md)
- July 2021 - Released PLBART checkpoints fine-tuned on downstream tasks
- June 2021 - Official code release
- March 2021 - Pre-release of source code


### Setup (optional)

We can setup a conda environment in order to run PLBART experiments, the first step is to download the dependencies. 
We assume [anaconda](https://www.anaconda.com/) and Python 3.6 is installed. The additional requirements 
(as noted in [requirements.txt](https://github.com/wasiahmad/PLBART/blob/main/requirements.txt) can be installed by 
running the following script:

```
bash install_tools.sh
```


### Pre-training

Install [apex](https://github.com/nvidia/apex#quick-start) for fp16 training. Then, follow the following steps.

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


### Fine-tuning on Downstream Tasks

We fine-tune and evaluate PLBART on three types of tasks.

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Task</th>
            <th>Language(s)</th>
            <th>Data</th>
            <th>Scripts</th>
            <th>Checkpoints</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Code to Text</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text" target="_blank">Code summarization</a></td>
            <td>Python, Java, Ruby, <br> PHP, Javascript, Go</td>
            <td><a href="https://drive.google.com/file/d/1m1IvGgPhDBg-SL-LajtFGTLyAJVbD0i3" target="_blank">[LINK]</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_text">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/1z_xC4-k8liAT1ir6r75sGza5BIXhzwUU" target="_blank">[LINK]</a></td>
        </tr>
        <tr>
            <td>Text to Code</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code" target="_blank">Code generation</a></td>
            <td>Java</td>
            <td><a href="https://drive.google.com/file/d/1rQjQh4Mle3yYzQbn-CRs4L1moZaAqr90" target="_blank">[LINK]</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/text_to_code">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/1Yk6YjoBELcKLFp8IyLF0-YfNQIY9NAqH" target="_blank">[LINK]</a></td>
        </tr>
        <tr>
            <td rowspan=4>Code to Code</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans" target="_blank">Code translation</a></td>
            <td>Java, C#</td>
            <td rowspan=4><a href="https://drive.google.com/file/d/15jokCxFQ9BUbptMsrfj4RdH_KiNkTRP2" target="_blank">[LINK]</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/translation">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/1KKdBWTRjnxC70icQrCbCXuj6ahMFQlE0" target="_blank">[LINK]</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement" target="_blank">Code refinement</a></td>
            <td>Java</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/refinement">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/19YYUvTnZbWeY064fZ165mmS4QHOxuYnC" target="_blank">[LINK]</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench" target="_blank">Clone detection</a></td>
            <td>Java</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/clone_detection">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/1bbjrvd_-etkJ1Za3fqv5Ea59vH3Wrf_1" target="_blank">[LINK]</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection" target="_blank">Defect detection</a></td>
            <td>C/C++</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/defect_prediction">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/1_YtIeBY2rLH-ICU1GsK7rmmt_ocxX4bd" target="_blank">[LINK]</a></td>
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

#### Step3. Build parser for CodeBLEU evaluation

```bash
cd evaluation/CodeBLEU/parser
bash build.sh
cd ../../..
```

#### Step4. Prepare the data, train and evaluate PLBART

For example, we want to fine-tune PLBART on `Text-to-Code` task. Then,

```bash
cd scripts/text_to_code
bash prepare.sh
bash run.sh GPU_IDS
cd ../..
```

Note. We fine-tuned PLBART on 1 `GeForce RTX 2080` (11gb) GPU.


### FAQ

__`mbart_base` task is not present in `fairseq==0.9.0` official release.__

Although we used `fairseq==0.9.0` but we used a different commit which consists of `mbart_base` task. You may do the 
following which should work.

```
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout 698e3b91ffa832c286c48035bdff78238b0de8ae
pip install .
```

Otherwise, you may consider installing `fairseq==0.10.0`. Please refer to this [issue](https://github.com/wasiahmad/PLBART/issues/12#issuecomment-881332837) 
to make other adjustments.

__What can be the maximum input and output lengths for PLBART?__

The maximum length is 512.


### Acknowledgement

PLBART uses [Fairseq](https://github.com/pytorch/fairseq), [codeXglue](https://github.com/microsoft/CodeXGLUE), and [TransCoder](https://github.com/facebookresearch/TransCoder) and thanks the authors of these works for their contribution.


### Citation

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

