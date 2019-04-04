LINSPECTOR
===========
![text](https://i.ibb.co/tb0psXM/Copy-of-Inspector-Gadget.png)
Language Inspector (LINSPECTOR) is an effort to interpret the **multilingual** black box NLP models. With this work, we release an easy-to-use framework to help researchers,_especially the ones that are interested in world languages_, understand their word representations better. 

* We have created and released 15 probing tasks for 24 languages. 
    * If you only want to download the probing datasets, go to `intrinsic/data`
    * If you want to create your own probing tasks for other languages or with different settings, see [data_split/README](data_split/readme.md)
    * If you want to evaluate your new embeddings, or the intermediate representations extracted from a black box model, e.g., neural dependency parser, see the instructions in [intrinsic/evaluation/README](intrinsic/evaluation/README.md)

* We have compiled and preprocessed (when necessary) dataset for universal part-of-speech tagging, dependency parsing, semantic role labeling, natural language inference and named entity recognition for Turkish, Finnish, German, Spanish and Russian. We provide the necessary data splits (when available) or guide you how to process the downloaded files with our preprocessing scripts.
    * If you want to evaluate your new embeddings on the downstream tasks, we provide config files that are ready to run with AllenNLP. 
    * Check [extrinsic/README](extrinsic/readme.md) for more details. 

Cite
-------
Please use the following citation:
```
@article{DBLP:journals/corr/abs-1903-09442,
  author    = {G{\"{o}}zde G{\"{u}}l {\C{S}}ahin and
               Clara Vania and
               Ilia Kuznetsov and
               Iryna Gurevych},
  title     = {{LINSPECTOR:} Multilingual Probing Tasks for Word Representations},
  journal   = {CoRR},
  volume    = {abs/1903.09442},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.09442},
  archivePrefix = {arXiv},
  eprint    = {1903.09442},
  timestamp = {Mon, 01 Apr 2019 14:07:37 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-09442},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

> **Abstract:** 
Despite an ever growing number of word representation models introduced for a large number of languages, there is a lack of a standardized technique to provide insights into what is captured by these models. Such insights would help the community to get an estimate of the downstream task performance, as well as to design more informed neural architectures, while avoiding extensive experimentation which requires substantial computational resources not all researchers have access to. A recent development in NLP is to use simple classification tasks, also called probing tasks, that test for a single linguistic feature such as part-of-speech. Existing studies mostly focus on exploring the information encoded by the sentence-level representations for English. However, from a typological perspective the morphologically poor English is rather an outlier: the information encoded by the word order and function words in English is often stored on a subword, morphological level in other languages. To address this, we introduce 15 word-level probing tasks such as case marking, possession, word length, morphological tag count and pseudoword identification for 24 languages. We present experiments on several state of the art word embedding models, in which we relate the probing task performance for a diverse set of languages to a range of classic NLP tasks such as semantic role labeling and natural language inference. We find that a number of probing tests have significantly high positive correlation to the downstream tasks, especially for morphologically rich languages. We show that our tests can be used to explore word embeddings or black-box neural models for linguistic cues in a multilingual setting. 

Contact person:

Gözde Gül Şahin, sahin@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

Please send us an e-mail or report an issue, if something is broken or if you have further questions.
       

       
