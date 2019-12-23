# allennlp-subword-eval


An implementation of a diagnostic classifier to evaluate word embeddings using AllenNLP framework. The model uses a two-layers feedforward classifier to predict a particular linguistic feature of a given word vector. The code is heavily adapted from AllenNLP tutorial ['Predicting Paper Venues'](https://github.com/allenai/allennlp/tree/master/tutorials#a-complete-example-predicting-paper-venues).


### Installations

1. Prepare a clean conda environment using Python 3.6.

2. Install some dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the embedding files to `static_embeddings` and `static_context_embeddings` (see below). You can just copy static word2vec (or muse) files to `static_embeddings` or create those files yourself (see `../../data_util`)


### Training a Model

To train a model, we need to prepare the data first. This repository includes some example data, i.e., see 'dataset' folder. The input of the model consists of two files:

1. A word embeddings file, ends with `.vec`, which has format word followed by the embeddings, all separated by a white space. Example:

```bash
running 0.11 0.23 ... 0.12
```

2. A label file, ends with `.txt`, which has format word followed by the target label, separated by a tab. Example:

```bash
running	VERB
```

You can use the provided scripts to prepare your data, currently we have script for [fastText](https://github.com/gozdesahin/dataset_compilation/blob/master/data_util/extract_fastText.py) and [word2vec](https://github.com/gozdesahin/dataset_compilation/blob/master/data_util/extract_word2vec.py). Just replace some lines according to the files in your path (see the in-line comments.


Then, run the following command to train a model:

```bash
allennlp train \
    subwordclassifier.json \
    -s /output_dir \
    --include-package classifiers
```

You can adjust `scripts/subwordclassification.json` file to change the hyperparameters of the model.


### Training a model on a contextual probing task

To train a model, we need to prepare the data first. This repository includes some example data, i.e., see 'dataset' folder. The input of the model consists of two files:

#### Contextual embeddings
1. A word embeddings file, ends with `.vec`. Format: `Sentence TAB index of word TAB vector for the word` Example:

```bash
The dog is running.     1    0.11 0.23 ... 0.12
```

2. A label file, ends with `.txt`. Format: `Sentence TAB word index TAB label for the word`  Example:

```bash
The dog is running.     1   NOUN
```

#### Static embeddings
1. A word embeddings file, ends with `.vec`, which has format word followed by the embeddings, all separated by a white space. Example:

```bash
running 0.11 0.23 ... 0.12
```

2. A label file, ends with `.txt`, which has format word followed by the target label, separated by a tab. Example:

```bash
running	VERB
```

Please copy the embedding files to `../static_embeddings` or `../static_context_embeddings`. You can copy w2v (or muse) files directly to the static directory.

You can use the provided scripts to prepare your data, currently we have script for [fastText](https://github.com/gozdesahin/dataset_compilation/blob/master/data_util/extract_fastText.py) and [word2vec](https://github.com/gozdesahin/dataset_compilation/blob/master/data_util/extract_word2vec.py). Just replace some lines according to the files in your path (see the in-line comments.


Then, run the following command to train a model:

```bash
allennlp train \
    config.json \
    -s /output_dir \
    --include-package classifiers
```

You have to create the config.json yourself or use our script to create the config files.


### Evaluating the Model

To evaluate the model, we can run a similar script:

```bash
allennlp evaluate \
    output_dir/model.tar.gz \
    --include-package classifiers \
    dataset/test
```

### Prediction

To output prediction:

```bash
allennlp predict model_dir/model.tar.gz test_file_path \
	--use-dataset-reader \
	--predictor word_classifier \
	--output-file out.txt
```


### Training script

For your convenience, we have prepared a script to automatically train models for a number of languages, embeddings, and features. You can run the following command in your terminal:
```bash
python scripts/prepare_scripts.py --data_dir probing_data_dir --embedding_dir your_embedding_dir --embedding_types w2v,bpe --mode classification
```

Here, `probing_data_dir` is the path to the intrinsic data directory. `your_embedding_dir` should have the structure of `embeddings_dir/language_id/embedding_type/final_embeds.vec`. Embedding types include word2vec, bpe, fasttext, elmo, and muse (all lowercased, separated by a comma). Mode options are `classification` or `contrastive` (for paired word test). These four parameters are required to run the script. Other optional parameters can be seen in the `prepare_scripts.py` file. 

The script will generate three folders (these name can be set by adding more arguments, see the python script for details): 

1. `json_configs`: all configuration files needed for training

2. `bash_scripts`: bash script to train models

3. `probing_models`: where the trained models are stored

You can then start training by:
```bash
sh bash_scripts/run_classification_word2vec.sh
```

To output predictions, you can adapt/modify the `predict.py` script.


### Training script for contextual probing
For your convenience, we have prepared a script to automatically train models for a number of languages, embeddings, and features. You can run the following command in your terminal:
```bash
python prepare_contextual_scripts.py
```
The script is stored at `scripts/prepare_contextual_scripts.py`


You can then start training by:
```bash
sh train.sh
```