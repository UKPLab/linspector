Extrinsic Data
---------------
* **NER** data is compiled from various resources (please see the paper for details). We preprocess all datasets to have a unified CoNLL-03 format and split into train,dev,test files. For each langauge, we provide the training, dev and test split used in our experiments. Please cite the original providers of the datasets if you use them in your experiments. 
* **SRL** is not available due to licensing issues. For Turkish SRL data, please request `Turkish PropBank Original` from [this link](http://tools.nlp.itu.edu.tr/Datasets). Finnish data can be downloaded from [here](https://turkunlp.org/Finnish_PropBank/). Other SRL datasets are available from LDC, with catalog number _LDC2012T03_.
* **POS** is derived from Universal Dependency Treebanks. Preprocessed for convenience.
* **UD** is version 2.3 downloaded from [the project's website](https://universaldependencies.org/). We use Finnish-TDT, German-GSD, Russian-SynTagRus, Spanish-AnCora and Turkish-IMST banks. 
* **XNLI** data is downloaded from [the project's website](https://www.nyu.edu/projects/bowman/xnli/). You also need the translated files: "XNLI-MT 1.0.zip". 

**Preprocessing:**
* All necessary preprocessing files are provided under the `data/%TASKNAME`.


Extrinsic Configs
---------------
Install [allennlp](https://github.com/allenai/allennlp#installing-via-pip) for POS, DEP, NER and XNLI. 
For SRL, clone [this repo](https://github.com/gozdesahin/Subword_Semantic_Role_Labeling) and follow the instructions.


* All json configuration files are provided under the `config/%TASKNAME` except from SRL.
* To train and evaluate the models with AllenNLP, run `python config/%TASKNAME/run_%TASKNAME.py` (you might need to adjust paths)
 
