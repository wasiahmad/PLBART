#### Files in this Repository

```yaml
.
├── FILEs.md
├── LICENSE
├── README.md
├── data
│   ├── codeXglue
│   │   └── download.sh
│   ├── github
│   │   ├── README.md
│   │   └── preprocessing
│   │       ├── __init__.py
│   │       ├── detokenize.py
│   │       ├── preprocess.py
│   │       └── src
│   │           ├── __init__.py
│   │           ├── code_tokenizer.py
│   │           ├── dataset.py
│   │           ├── javalang_tokenizer.py
│   │           ├── timeout.py
│   │           └── utils.py
│   └── stackoverflow
│       ├── README.md
│       └── preprocess.py
├── evaluation
│   ├── CodeBLEU
│   │   ├── bleu.py
│   │   ├── calc_code_bleu.py
│   │   ├── dataflow_match.py
│   │   ├── keywords
│   │   │   ├── c_sharp.txt
│   │   │   ├── java.txt
│   │   │   └── python.txt
│   │   ├── parser
│   │   │   ├── DFG.py
│   │   │   ├── __init__.py
│   │   │   ├── build.py
│   │   │   ├── build.sh
│   │   │   ├── my-languages.so
│   │   │   └── utils.py
│   │   ├── readme.txt
│   │   ├── syntax_match.py
│   │   ├── utils.py
│   │   └── weighted_ngram_match.py
│   ├── bleu.py
│   └── evaluator.py
├── pretrain
│   ├── absolute.sh
│   ├── binarize.sh
│   └── download.sh
├── requirements.txt
├── scripts
│   ├── code_to_code
│   │   ├── clone_detection
│   │   │   ├── encode.py
│   │   │   ├── eval.py
│   │   │   ├── prepare.sh
│   │   │   └── run.sh
│   │   ├── defect_prediction
│   │   │   ├── encode.py
│   │   │   ├── eval.py
│   │   │   ├── prepare.sh
│   │   │   └── run.sh
│   │   ├── refinement
│   │   │   ├── encode.py
│   │   │   ├── prepare.sh
│   │   │   └── run.sh
│   │   └── translation
│   │       ├── encode.py
│   │       ├── prepare.sh
│   │       └── run.sh
│   ├── code_to_text
│   │   ├── encode.py
│   │   ├── evaluator.py
│   │   ├── prepare.sh
│   │   └── run.sh
│   └── text_to_code
│       ├── encode.py
│       ├── evaluator.py
│       ├── prepare.sh
│       └── run.sh
├── sentencepiece
│   ├── README.md
│   ├── dict.txt
│   ├── encode.py
│   ├── sentencepiece.bpe.model
│   ├── sentencepiece.bpe.vocab
│   └── train.py
├── setup.py
└── source
    ├── __init__.py
    ├── sentence_prediction.py
    └── translation.py

21 directories, 70 files
```