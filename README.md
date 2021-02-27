# ASCArgumentationMining2021
Creating an argument mining pipeline using the Atheism Stance Corpus (Wojatzki &amp; Zesch, 2016).

# Description of data
n = 715

Value counts for selected topics:

| topic | none | favor | against | mean |
|:---:|:---:|:---:|:---:|:---:|
| atheism | 220 | 147 | 348 | -0.281 |
| supernatural | 379 | 282 | 54 | 0.319 |
| christianity | 491 | 192 | 32 | 0.224 |
| islam | 672 | 29 | 14 | 0.021 |

Other topics have severely imbalanced classes.


# Installation
Developed in Python 3.7.3 under Twister OS 1.9.7 (Raspbian 32-bit).

The Atheism Stance Corpus can be obtained from [GitHub](https://github.com/muchafel/AtheismStanceCorpus). Please run ```ascpreprocess.py``` before working with the corpus to fix some formatting errors.

# Usage
## Training mode
```cmd
python3 main.py data target model -b -t -g
```

### Arguments
* ```data```: Path to the training data (any suitably formatted text file; ```ascpreprocess.py``` creates a .txt file). Needs column "text".
* ```target```: The training target (column in the corpus = topic), e.g. "atheism" or "religious_freedom".
* ```model```: Where the trained model should be pickled, e.g. .pkl file.
* ```-b```: *Optional* flag that trains a baseline model according to Mohammad et al. (2016), i.e. a linear SVM with word and character ngram features.
* ```-t```: *Optional* flag that enables test mode. 20% of the data will be held out for accuracy and micro f1 evaluation.
* ```-g```: *Optional* flag that enables sklearn's grid search. Cannot be used with ```-b```. May take a long time to compute on a large search space; you might want to adjust the transformers list or number of cross-validation folds in the code first.

### Examples
#### Just training
(You may adjust the model parameters in the ASCClassifier constructor.)
```cmd
python3 main.py data/corpus.txt atheism model.pkl
```

#### Training and testing the baseline model
```cmd
python3 main.py data/corpus.txt atheism model.pkl -b -t
```

## Prediction mode
```cmd
python3 main.py data model [output] -p
```

### Arguments
* ```data```: Path to the input data (text file).
* ```model```: Path to a pickled model, e.g. .pkl file.
* ```output```: *Optional* path where the predictions should be saved (tab-separated text file). If not supplied, the list of predictions is printed to the console.

### Examples
#### Writing to file
```cmd
python3 main.py data/my_data.txt model.pkl predictions.tsv -p
```

#### Printing to console
```cmd
python3 main.py data/my_data.txt model.pkl -p
```

# Benchmarks
All tests were done using ```random_state=0``` for splitting the training data.
| features | topic | accuracy | cv mean | cv std || topic | accuracy | cv mean | cv std |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **baseline** | *atheism* | 0.727 | 0.715 | **0.031** || *supern.* | 0.797 | 0.807 | 0.034 |
| **vanilla BOW** | *atheism* | 0.727 | 0.687 | 0.066 || *supern.* | 0.797 | 0.815 | 0.035 |
| **best BOW** | *atheism* | 0.762 | 0.734 | 0.039 || *supern.* | 0.811** | 0.816 | **0.029** |
| &#8627; parameters | *atheism* | ```min_df=```<br>```0.01``` | ```max_df=```<br>```0.7``` ||| *supern.* | none
| **best** | *atheism* | **0.818** | **0.757** | 0.058 || *supern.* | **0.832** | **0.820** | 0.037 |

&nbsp;
| features | topic | accuracy | cv mean | cv std || topic | accuracy | cv mean | cv std |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **baseline** | *christ.* | 0.811 | 0.843 | 0.027 || *islam* | 0.944 | 0.953 | 0.011 |
| **vanilla BOW** | *christ.* | 0.818 | 0.820 | 0.037 || *islam* | 0.937 | 0.958 | 0.016 |
| **best BOW** | *christ.* | 0.888 | 0.850 | **0.024** || *islam* | **0.951** | **0.962** | **0.010** |
| &#8627;parameters | *christ.* | ```min_df=```<br>```0.003``` | ```max_df=```<br>```0.7``` | ```ngram_range=```<br>```(1,2)``` || *islam* | none
| **best** | *christ.* | **0.902** | **0.855** | 0.029 || *islam* | 0.951 | 0.962 | 0.013 |
| **just polarity** | *christ.* | 0.713 | 0.699 | 0.036 ||

&nbsp;
* Vanilla BOW means sklearn's out-of-the-box ```CountVectorizer()``` with no parameters.
* Best BOW means custom preprocessing + best parameters for CountVectorizer() for this topic as found by grid search, but no other features.
* Best means the highest-scoring feature configuration, i.e. preprocessing + best BOW + pos_tfidf, polarity and word features. This is the default configuration.