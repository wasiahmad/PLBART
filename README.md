# PLBART
Code pre-release of our work, [Unified Pre-training for Program Understanding and Generation]() accepted at NAACL 2021.

**Note. [A detailed documentation is coming soon.]()**

### Pre-training data

PLBART is pre-trained on Java and Python functions and natural language descriptions collected from Github and StackOverflow.


### Evaluation tasks

We evaluated PLBART on five tasks.

- Code summarization [[REF](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text#dataset)]
- Code generation [[REF](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code#task-definition)]
- Code translation [[REF](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans#task-definition)]
- Clone detection [[REF](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench#task-definition)]
- Vulnerability REF [[REF](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection#codexglue----defect-detection)]


### Notes

- We will publish the pretrained PLBART checkpoint soon.
- We list all the files in this repository [here](https://github.com/plbart-2020/PLBART/blob/main/FILEs.md).

### Acknowledgement

PLBART uses [Fairseq](https://github.com/pytorch/fairseq), [codeXglue](https://github.com/microsoft/CodeXGLUE), and [TransCoder](https://github.com/facebookresearch/TransCoder) and thanks the authors of these works for their contribution.


### Citation

```
@inproceedings{ahmad2020summarization,
    author = {Ahmad, Wasi Uddin and Chakraborty, Saikat and Ray, Baishakhi and Chang, Kai-Wei},
    booktitle = {Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics},
    title = {Unified Pre-training for Program Understanding and Generation},
    year = {2021}
}
```

