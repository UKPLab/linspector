# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-01-21 11:26:19
# @Last Modified by:   claravania
# @Last Modified time: 2019-03-20 10:43:44


import os
import sys

from collections import defaultdict

# mode is either classification or contrastive
model_mode = 'classification'
embeddings = ['word2vec', 'muse', 'fasttext', 'elmo', 'bpe']


feature_set = model_mode + "_features.txt"

feat_dict = defaultdict(list)
with open(feature_set, encoding='utf-8') as f:
	for line in f:
		line = line.strip()
		feature, lang, _, _ = line.split('\t')
		feature = feature.replace(' ', '_')
		feat_dict[feature].append(lang)


for emb_type in embeddings:
	scripts_dir = "scripts/" + model_mode + '/' + emb_type + '/'
	bash_file = 'run_' + model_mode + '_' + emb_type + '.sh'
	with open(bash_file, 'w', encoding='utf-8') as f:
		f.write('#!/bin/bash\n\n')
		f.write('export PYTHON_PATH=$PATH\nexport PYTHONIOENCODING=utf-8\n\n')
		f.write('source ~/.bashrc\nconda activate subwordeval\n\n')
		for feature in feat_dict.keys():
			for lang in feat_dict[feature]:
				json_file = feature + '-' + lang + '.json'
				f.write('allennlp train ' + scripts_dir + json_file + ' -s models/' + lang + '/' + emb_type + '/' + feature + ' --include-package classifiers\n')


	bash_file = 'evaluate_' + model_mode + '_' + emb_type + '.sh'
	with open(bash_file, 'w', encoding='utf-8') as f:
		f.write('#!/bin/bash\n\n')
		f.write('export PYTHON_PATH=$PATH\nexport PYTHONIOENCODING=utf-8\n\n')
		f.write('source ~/.bashrc\nconda activate subwordeval\n\n')
		for feature in feat_dict.keys():
			for lang in feat_dict[feature]:
				output_file = os.path.join('outputs', lang, emb_type, feature + '.json')
				f.write('allennlp evaluate models/'  + lang + '/' + emb_type + '/' + feature + '/model.tar.gz --include-package classifiers ../../dataset_compilation/final_tests/' + feature + '/' + lang + '/test.txt --output-file ' +  output_file + '\n')
			f.write('\n')
