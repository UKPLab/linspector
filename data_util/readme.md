data_util:
=========
This package contains readers/writers and preprocessors for the data used in our extrinsic and intrinsic experiments.

Unimorph and Sigmorphon utility: 
-------------------------------
* `reader.load_ds(ds)` - loads **dataset** _ds_. See dataset names in _global_var.py_
    * A **dataset** is a collection of dictionaries {"form":..., "lemma":..., "msd":...} grouped by language
    * MSD is the Unimorph morphological code. It can be converted to feature-value representation by calling `schema.UnimorphSchema.decode_msd(msd)`


Word embedding utility:
----------------------
**Requirements:**
* fastText - to install, follow the instructions for 'Building fastText for Python' from https://github.com/facebookresearch/fastText
* Gensim - version 3.7 or later
* BPE - `pip install bpemb`
* AllenNLP - `pip install allennlp` 
* ELMO - `python setup.py install` after cloning the repo from https://github.com/HIT-SCIR/ELMoForManyLangs

You also need to download fastText and ELMO embeddings for the languages you are interested in, before you proceed.

**Available Functions:**

`get_intrinsic_vocab` - Extracts and saves the vocabulary of the probing tasks for given languages

`get_extrinsic_vocab` - Extracts and saves the vocabulary of the downstream tasks for given languages. We currently support Universal Dependency Treebanks, CoNLL-09 Semantic Role Labeling datasets (also Finnish PropBank), Named Entity Recognition datasets and Cross-lingual Natural Language Inference (X-NLI)    

`extract_vectors` - Given a pretrained word embedding file and a set of words, it extracts the vectors in batches and save them in embeds.vec file. We currently support: word2vec, fastText, BPE, ELMO and MUSE.


**Pipeline:**

We use the following pipeline in our experiments:

(1) Extract intrinsic and extrinsic vocabularies with `get_intrinsic_vocab.py` and `get_extrinsic_vocab.py`. 

(2) Save the embeddings for the words in the vocabulary (See the scripts: `extract_vectors_intrinsic.sh` and `extract_vectors_extrinsic.sh`). 

(3) Merge the vocabulary with vector file (see the scripts `merge_files_intrinsic.sh` and `merge_files_extrinsic.sh`)   
