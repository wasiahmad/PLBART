#### Files in this Repository

```
.
├── LICENSE
├── README.md
├── codeXglue
│   ├── code_to_code
│   │   ├── CodeBLEU
│   │   │   ├── bleu.py
│   │   │   ├── calc_code_bleu.py
│   │   │   ├── dataflow_match.py
│   │   │   ├── keywords
│   │   │   │   ├── c_sharp.txt
│   │   │   │   ├── java.txt
│   │   │   │   └── python.txt
│   │   │   ├── parser
│   │   │   │   ├── DFG.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── build.py
│   │   │   │   ├── build.sh
│   │   │   │   ├── my-languages.so
│   │   │   │   └── utils.py
│   │   │   ├── readme.txt
│   │   │   ├── syntax_match.py
│   │   │   ├── utils.py
│   │   │   └── weighted_ngram_match.py
│   │   ├── bleu.py
│   │   ├── clone_detection
│   │   │   ├── encode.py
│   │   │   ├── eval.py
│   │   │   ├── evaluator.py
│   │   │   ├── prepare.sh
│   │   │   └── run.sh
│   │   ├── defect_prediction
│   │   │   ├── encode.py
│   │   │   ├── eval.py
│   │   │   ├── evaluator.py
│   │   │   ├── prepare.sh
│   │   │   └── run.sh
│   │   ├── encode.py
│   │   ├── evaluator.py
│   │   ├── refin_prep.sh
│   │   ├── refin_run.sh
│   │   ├── trans_prep.sh
│   │   └── trans_run.sh
│   ├── code_to_text
│   │   ├── download.sh
│   │   ├── encode.py
│   │   ├── evaluator.py
│   │   ├── generate.sh
│   │   ├── multilingual.sh
│   │   ├── prep.sh
│   │   ├── python_tokenizer.py
│   │   └── run.sh
│   └── text_to_code
│       ├── CodeBLEU
│       │   ├── bleu.py
│       │   ├── calc_code_bleu.py
│       │   ├── dataflow_match.py
│       │   ├── keywords
│       │   │   ├── c_sharp.txt
│       │   │   ├── java.txt
│       │   │   └── python.txt
│       │   ├── parser
│       │   │   ├── DFG.py
│       │   │   ├── __init__.py
│       │   │   ├── build.py
│       │   │   ├── build.sh
│       │   │   ├── my-languages.so
│       │   │   └── utils.py
│       │   ├── readme.txt
│       │   ├── syntax_match.py
│       │   ├── utils.py
│       │   └── weighted_ngram_match.py
│       ├── bleu.py
│       ├── encode.py
│       ├── evaluator.py
│       ├── generate.sh
│       ├── prep.sh
│       └── run.sh
├── preprocessing
│   ├── __init__.py
│   ├── detokenize.py
│   ├── preprocess.py
│   ├── src
│   │   ├── __init__.py
│   │   ├── code_tokenizer.py
│   │   ├── dataset.py
│   │   ├── javalang_tokenizer.py
│   │   ├── test_tokenize_cpp.py
│   │   ├── test_tokenize_java.py
│   │   ├── test_tokenize_python.py
│   │   ├── timeout.py
│   │   └── utils.py
│   └── test_preprocess.py
├── pretrain
│   ├── absolute.sh
│   └── binarize.sh
├── requirements.txt
├── sentencepiece
│   ├── encode.py
│   └── train.py
├── setup.py
└── stackoverflow
    └── preprocess.py
```