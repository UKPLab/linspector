# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-02-19 15:15:05
# @Last Modified by:   claravania
# @Last Modified time: 2019-04-01 14:54:02

import sys
import os

from allennlp.commands import main
from collections import defaultdict

# mode is either classification or contrastive
mode = 'classification'
# embedding types
embeddings = ['word2vec', 'muse', 'fasttext', 'elmo', 'bpe']
# test data directory
data_dir = "/afs/inf.ed.ac.uk/group/project/datacdt/s1459234/projects/subword_probers/dataset_compilation/final_tests"

feature_set = "scripts/" + mode + "_features.txt"
# languages to test
langId = {'finnish': 'fi',
          'russian': 'ru',
          'german': 'de',
          'spanish': 'es',
          'turkish': 'tr'}

feat_dict = defaultdict(lambda: defaultdict(int))
with open(feature_set) as f:
    for line in f:
        line = line.strip()
        feature, lang, count, _ = line.split('\t')
        feature = feature.replace(' ', '_')
        feat_dict[feature][lang] = count

for emb_types in embeddings:
    for task in feat_dict.keys():
        for lang in feat_dict[task]:
            model_dir = os.path.join("models", lang, emb_types, task, "model.tar.gz")
            input_file = os.path.join(data_dir, task, lang, 'test.txt')
            output_file = os.path.join('predictions', lang, emb_types, task + '.txt')
            predictor = 'word-classifier'
            package = 'classifiers'

            sys.argv = [
                "allennlp",  # command name, not used by main
                "predict",
                model_dir,
                input_file,
                "--use-dataset-reader",
                "--predictor", predictor,
                "--include-package", package,
                "--output-file", output_file
            ]

            main()
