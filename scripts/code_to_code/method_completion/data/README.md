# Github Python Data

A recent effort in training GPT-2 model called [CodeParrot](https://huggingface.co/blog/codeparrot) that can
auto-complete Python code, the author shared the pre-training dataset. The dataset is constructed from the GitHub dump
available on Google's BigQuery and filtered for all Python files. The result is a **180 GB** dataset with
**20 million** files (available [here](https://huggingface.co/datasets/transformersbook/codeparrot)). They also cleaned
the dataset and made it available on the Hugging Face Hub (
available [here](https://huggingface.co/datasets/lvwerra/codeparrot-clean)).

We use this dataset to train PLBART to learn to auto-complete Python functions. In the following section, we discuss the
Dataset preparation steps.

## Data Preparation

### Download, Processing and Function Extraction, Binarization

``` 
cd github
bash setup.sh
```

Once finished, in the saved directory (`./python/download`), the following files will appear.

``` 
file-*.with_comments.json.gz
file-*.with_comments.tok
test.with_comments.functions_class.tok
test.with_comments.functions_standalone.tok
test.with_comments.tok
train.with_comments.[0-7].functions_class.tok
train.with_comments.[0-7].functions_standalone.tok
train.with_comments.[0-7].tok
valid.with_comments.functions_class.tok
valid.with_comments.functions_standalone.tok
valid.with_comments.tok
```

The sentencepiece tokenized and binarized data will appear at `./python/spm` and `./python/shards`
directories, respectively.

### Statistics

- Total documents (.py files) - 5,338,968
- Total functions - 61,682,262
    - Total standalone functions - 15,918,155
    - Total class functions - 45,764,107
- Total code tokens - 10,375,418,020

## Unlabeled Function to Source-Target Pairs

An example line from `train.with_comments.[0-7].functions_class.tok` file:

``` 
def test_plugin_inheritance ( self ) : NEW_LINE INDENT """ Test ▁ that ▁ an ▁ object ▁ derived ▁ from ▁ BasePlugin ▁ works ▁ properly """ NEW_LINE simple_plugin = self . SimplePlugin ( ) NEW_LINE self . assertEqual ( simple_plugin . routes ( ) , [ ] ) NEW_LINE DEDENT
```

If we detokenize the example, we see:

``` 
def test_plugin_inheritance ( self ) :
    """ Test that an object derived from BasePlugin works properly """
    simple_plugin = self.SimplePlugin ( )
    self.assertEqual ( simple_plugin.routes ( ) , [ ] )
```

We form training examples (**<source, target>**) to train PLBART following two strategires.

### Strategy 1 - Random

We construct examples (**<source, target>**) by randomly splitting functions.

``` 
# Source
def test_plugin_inheritance ( self ) :
    """ Test that an object derived 
    
# Target
    from BasePlugin works properly """
    simple_plugin = self.SimplePlugin ( )
    self.assertEqual ( simple_plugin.routes ( ) , [ ] )
```

### Strategy 2 - Eval-Aligned

With 25% probability, we construct examples as:

``` 
# Source (function signature + docstring)
def test_plugin_inheritance ( self ) :
    """ Test that an object derived from BasePlugin works properly """
    
# Target (function body)
    simple_plugin = self.SimplePlugin ( )
    self.assertEqual ( simple_plugin.routes ( ) , [ ] )
```

With 25% probability, we construct examples as:

``` 
# Source (function signature)
def test_plugin_inheritance ( self ) :
    
# Target (function body)
    """ Test that an object derived from BasePlugin works properly """
    simple_plugin = self.SimplePlugin ( )
    self.assertEqual ( simple_plugin.routes ( ) , [ ] )
```

With 25% probability, we construct examples as:

``` 
# Source (docstring)
Test that an object derived from BasePlugin works properly

# Target (whole function)
def test_plugin_inheritance ( self ) :
    simple_plugin = self.SimplePlugin ( )
    self.assertEqual ( simple_plugin.routes ( ) , [ ] )
```

With 25% probability, we construct examples as:

``` 
# Source (random split)
def test_plugin_inheritance ( self ) :
    """ Test that an object derived from BasePlugin works properly """
    simple_plugin = 

# Target (function \ Source)
                    self.SimplePlugin ( )
    self.assertEqual ( simple_plugin.routes ( ) , [ ] )
```


