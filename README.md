# [PLBART](https://www.aclweb.org/anthology/2021.naacl-main.211/)

Official code release of our NAACL 2021 work, [Unified Pre-training for Program Understanding and Generation](https://www.aclweb.org/anthology/2021.naacl-main.211/). 
We present **PLBART** that is pre-trained on a large collection Java and Python functions and natural language descriptions collected from Github and StackOverflow, respectively.

We present the file structure of this repository [here](https://github.com/wasiahmad/PLBART/blob/main/FILEs.md).


### [[Optional] Setup]()

We can setup a conda environment in order to run PLBART experiments, the first step is to download the dependencies. 
We assume `[anaconda](https://www.anaconda.com/)` and Python 3.6 is installed. The additional requirements 
(as noted in [requirements.txt](https://github.com/wasiahmad/PLBART/blob/main/requirements.txt) can be installed by 
running the following script:

```
bash install_tools.sh
```


### [Pre-training]()

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
            <th>Fine-tuned Checkpoints</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Code to Text</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text" target="_blank">Code summarization</a></td>
            <td>Python <br> Java <br> Ruby <br> PHP <br> Javascript <br> Go</td>
            <td><a href="https://drive.google.com/file/d/1m1IvGgPhDBg-SL-LajtFGTLyAJVbD0i3" target="_blank">code-to-text.zip</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_text">[LINK]</a></td>
            <td>
                <a href="https://drive.google.com/drive/folders/1ijkcsxINtxvKz_DDe7TYCUq15meizLcR" target="_blank">[python_en_XX]</a>
                <br>
                <a href="https://drive.google.com/drive/folders/1MNkQPGLdnhcP_MnouTLrlm1IyN65OAED" target="_blank">[java_en_XX]</a>
                <br>
                <a href="https://drive.google.com/drive/folders/18ix4BE6z8S_qa2uk6c2jnB7EheVvdP6Q" target="_blank">[ruby_en_XX]</a>
                <br>
                <a href="https://drive.google.com/drive/folders/1RbNXh0RCOejuH64nPAz3Y6H37JtrchcQ" target="_blank">[php_en_XX]</a>
                <br>
                <a href="https://drive.google.com/drive/folders/1-nu3395dxnUliOGQ44QT2KbE6YWoDXJj" target="_blank">[javascript_en_XX]</a>
                <br>
                <a href="https://drive.google.com/drive/folders/1ZImgzyaLbRJxVFw_AdSuVz_Eqn54N_ee" target="_blank">[go_en_XX]</a>
            </td>
        </tr>
        <tr>
            <td>Text to Code</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code" target="_blank">Code generation</a></td>
            <td>Java</td>
            <td><a href="https://drive.google.com/file/d/1rQjQh4Mle3yYzQbn-CRs4L1moZaAqr90" target="_blank">text-to-code.zip</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/text_to_code">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/11vdZ5cbT23-KII_4Rn1QvsJBEcV7PiuJ" target="_blank">[concode]</a></td>
        </tr>
        <tr>
            <td rowspan=4>Code-to-Code</td>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans" target="_blank">Code translation</a></td>
            <td>Java, C#</td>
            <td rowspan=4><a href="https://drive.google.com/file/d/15jokCxFQ9BUbptMsrfj4RdH_KiNkTRP2" target="_blank">code-to-code.zip</a></td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/translation">[LINK]</a></td>
            <td>
                <a href="https://drive.google.com/drive/folders/1OTibmDXrNXcg4an1ehSx1W-JNG5M2kRY" target="_blank">[java_cs]</a>
                <br>
                <a href="https://drive.google.com/drive/folders/1_TCQMfGhuS8-DcE-SII8sEbxryEWG3Zx" target="_blank">[cs_java]</a>
            </td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-refinement" target="_blank">Code refinement</a></td>
            <td>Java</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/refinement">[LINK]</a></td>
            <td>
                <a href="https://drive.google.com/drive/folders/16WgiNhYQkKm_oLJuqXddgnWlvsdS9i6Y" target="_blank">[small]</a>
                <br>
                <a href="https://drive.google.com/drive/folders/1jOaH7l0OtKIEQL0j-IGeYCZ_2mV07h9x" target="_blank">[medium]</a>
            </td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench" target="_blank">Clone detection</a></td>
            <td>Java</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/clone_detection">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/1WnlmmOi4py3bQsS0zSfXxRLoSue_Iv-J" target="_blank">[clone_detection]</a></td>
        </tr>
        <tr>
            <td><a href="https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection" target="_blank">Defect detection</a></td>
            <td>C/C++</td>
            <td><a href="https://github.com/wasiahmad/PLBART/tree/main/scripts/code_to_code/defect_prediction">[LINK]</a></td>
            <td><a href="https://drive.google.com/drive/folders/1zSAGkHKzlRFQjoZ2qTnkrRWrjGMDfzR0" target="_blank">[devign]</a></td>
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


### [FAQ]()

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

