# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-02-12 16:44:20
# @Last Modified by:   claravania
# @Last Modified time: 2019-03-20 10:42:11


import os
import json

from collections import defaultdict


# this script reads metrics.json files and outputs the summary of accuracy for each model

embeddings = ['word2vec', 'fasttext', 'bpe', 'elmo', 'muse']
# mode is either classification or contrastive
mode = "contrastive"
feature_set = mode + "_features.txt"

feat_dict = defaultdict(list)
with open(feature_set, encoding='utf-8') as f:
	for line in f:
		line = line.strip()
		feature, lang, _, _ = line.split('\t')
		feature = feature.replace(' ', '_')
		feat_dict[lang].append(feature)


for lang in sorted(feat_dict.keys()):
	print('Language:', lang)
	for task in sorted(feat_dict[lang]):
		items = [task]
		for emb_type in embeddings:
			output_file = os.path.join('../outputs', lang, emb_type, task + '.json')
			with open(output_file) as f:
				data = json.load(f)

			acc = round(float(data['accuracy']) * 100, 1)
			items.append(acc)
		print(' '.join([str(x) for x in items]))

