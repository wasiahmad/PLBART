# PLBART as Function Autocomplete

We train PLBART to learn to auto-complete Python functions.

## Setup

``` 
bash install_env.sh
conda activate plbart-fa
```

## Training

### Data

- CodeSearchNet (./data/csnet/setup.sh)
- GitHub-python available on Google's BigQuery (./data/github/setup.sh)
- Please read the README.md of the data folder.

### Run

``` 
# train plbart_base on GitHub-python data
bash completion.sh 0,1,2,3,4,5,6,7 base github
```

**[NOTE]**

- We found CodeSearchNet data not useful as the training data.
- We only explored the random data split strategy as discussed in README.md of the data folder.

## Evaluation

### [Human-Eval](https://github.com/openai/human-eval)

Human-Eval is a benchmark dataset released by OpenAI. Every example in the dataset is a JSON object that has the
following keys.

```
"task_id": "HumanEval/0", 
"prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", 
"entry_point": "has_close_elements", 
"canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", 
"test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
```

In Human-Eval, a model generated code is evaluated by executing:

``` 
prompt + completion + "\n" + test + "\n" + f"check({entry_point})"
```

For example, if we use the `canonical_solution` as completion, the above expression results in the following.

``` 
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False

METADATA = {
    'author': 'jt',
    'dataset': 'test'
}

def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
    
check(has_close_elements)
```

### Run

``` 
# evaluate plbart_base trained on Github-python
bash evaluate.sh 0 human-eval base
```

**[NOTE]** Modify `CKPT_DIR` and `CKPT_FILE` variables in the `evaluate.sh` script to test a different model.

### Results

Baseline results are collected from [here](https://huggingface.co/blog/codeparrot#evaluation).

<table>
    <thead>
        <tr>
            <th colspan=1 align ="left">Model</th>
            <th colspan=1">#Params</th>
            <th colspan=1>Pass@1</th>
            <th colspan=1>Pass@10</th>
            <th colspan=1>Pass@100</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>CodeParrot</td>
            <td align ="center">110M</td>
            <td align ="center">3.80%</td>
            <td align ="center">6.57%</td>
            <td align ="center">12.78%</td>
        </tr>
        <tr>
            <td align ="center">1.5B</td>
            <td align ="center">3.58%</td>
            <td align ="center">8.03%</td>
            <td align ="center">14.96%</td>
        </tr>
        <tr>
            <td rowspan=4>Codex</td>
            <td align ="center">25M</td>
            <td align ="center">3.21%</td>
            <td align ="center">7.1%</td>
            <td align ="center">12.89%</td>
        </tr>
        <tr>
            <td align ="center">85M</td>
            <td align ="center">8.22%</td>
            <td align ="center">12.81%</td>
            <td align ="center">22.40%</td>
        </tr>
        <tr>
            <td align ="center">300M</td>
            <td align ="center">13.17%</td>
            <td align ="center">20.37%</td>
            <td align ="center">36.27%</td>
        </tr>
        <tr>
            <td align ="center">12B</td>
            <td align ="center"><b>28.81%</b></td>
            <td align ="center"><b>46.81%</b></td>
            <td align ="center"><b>72.31%</b></td>
        </tr>
        <tr>
            <td rowspan=1>GPT-J</td>
            <td align ="center">6B</td>
            <td align ="center">11.62%</td>
            <td align ="center">15.74%</td>
            <td align ="center">27.74%</td>
        </tr>
        <tr>
            <td rowspan=3>GPT-Neo</td>
            <td align ="center">125M</td>
            <td align ="center">0.75%</td>
            <td align ="center">1.88%</td>
            <td align ="center">2.97%</td>
        </tr>
        <tr>
            <td align ="center">1.5B</td>
            <td align ="center">4.79%</td>
            <td align ="center">7.47%</td>
            <td align ="center">16.30%</td>
        </tr>
        <tr>
            <td align ="center">2.7B</td>
            <td align ="center">6.41%</td>
            <td align ="center">11.27%</td>
            <td align ="center">21.37%</td>
        </tr>
        <tr>
            <td rowspan=1>PLBART</td>
            <td align ="center">110M</td>
            <td align ="center">3.96%</td>
            <td align ="center">7.8%</td>
            <td align ="center">12.8%</td>
        </tr>
    </tbody>
</table>

**[Disclaimer]** We did not carefully tune the hyper-parameters.

## License

Please abide by the license of both the training and evaluation data.
