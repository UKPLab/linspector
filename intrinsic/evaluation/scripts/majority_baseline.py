# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-02-14 17:24:48
# @Last Modified by:   claravania
# @Last Modified time: 2019-03-20 10:42:24


import os
import codecs
import operator

from collections import defaultdict


# this script computes the majority baseline
# that is, accuracy if we predict the majority label for each task


def compute_label_dist(fname):
    label_dist = defaultdict(int)
    with codecs.open(fname, encoding='utf-8') as f:
        for line in f:
            _, _, label = line.strip().split('\t')
            label_dist[label] += 1
    return label_dist


# mode is either classification or contrastive
mode = 'classification'

feature_set = mode + "_features.txt"
feat_dict = defaultdict(list)
with codecs.open(feature_set) as f:
    for line in f:
        line = line.strip()
        feature, lang, _, _ = line.split('\t')
        feat_dict[lang].append(feature)

data_dir = '../../../dataset_compilation/final_tests'
for lang in sorted(feat_dict.keys()):
    print('Language:', lang)
    for feature in sorted(feat_dict[lang]):
        feature_data_dir = os.path.join(data_dir, feature, lang)
        train_file = os.path.join(feature_data_dir, 'train.txt')

        label_dist = compute_label_dist(train_file)
        sorted_label_dist = sorted(label_dist.items(), key=operator.itemgetter(1), reverse=True)
        majority_label = sorted_label_dist[0][0]

        test_file = os.path.join(feature_data_dir, 'test.txt')
        num_instances = 0
        num_true = 0
        with codecs.open(test_file, encoding='utf-8') as f:
            for line in f:
                _, _, label = line.strip().split('\t')
                num_instances += 1
                if label == majority_label:
                    num_true += 1

        acc = round(num_true * 100.0 / num_instances, 1)
        print(feature, majority_label.replace(' ', '_').lower(), acc)
    print()
