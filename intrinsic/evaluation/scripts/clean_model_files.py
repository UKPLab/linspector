# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-02-13 13:54:17
# @Last Modified by:   claravania
# @Last Modified time: 2019-03-20 10:40:17

import os

from collections import defaultdict

# the default config generate a model file for every epoch.
# this script deletes all model files except the one from the best epoch.


embeddings = ['word2vec', 'muse', 'fasttext', 'elmo', 'bpe']
mode = 'classification'

feature_set = mode + "_features.txt"
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
            model_dir = os.path.join('../models', lang, emb_types, task)
            for fname in os.listdir(model_dir):
                if fname.startswith('model_state_epoch') or fname.startswith('training_state_epoch'):
                    file_path = os.path.join(model_dir, fname)
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))
