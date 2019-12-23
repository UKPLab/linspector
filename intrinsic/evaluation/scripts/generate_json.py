# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-01-21 10:43:25
# @Last Modified by:   claravania
# @Last Modified time: 2019-03-20 10:44:42

import os

from collections import defaultdict

embeddings = ['word2vec', 'muse', 'fasttext', 'elmo', 'bpe']
# mode is either classification or contrastive
mode = 'contrastive'

json_template = "subword" + mode + ".json"
feature_set = mode + "_features.txt"
embeddings_dir = "/afs/inf.ed.ac.uk/group/project/datacdt/s1459234/projects/subword_probers/embeddings/intrinsic"
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
            fout = open(os.path.join(mode, emb_types, task + '-' + lang + '.json'), 'w')
            embedding_file = os.path.join(embeddings_dir, langId[lang], emb_types, 'final_embeds.vec')
            with open(json_template) as f:
                for line in f:
                    if '_data_path' in line:
                        pair = task + '/' + lang
                        newline = line.replace('[feat]/[lang]', pair)
                        fout.write(newline)
                    elif emb_types == 'elmo' and 'embedding_dim' in line:
                        newline = line.replace('300', '1024')
                        fout.write(newline)
                    elif emb_types == 'elmo' and 'input_dim' in line:
                        if mode == 'classification':
                            newline = line.replace('300', '1024')
                        elif mode == 'contrastive':
                            newline = line.replace('600', '2048')
                        fout.write(newline)
                    elif 'hidden_dims' in line:
                        newline = line.replace('x', str(feat_dict[task][lang]))
                        fout.write(newline)
                    elif 'pretrained_embedding_file' in line:
                        newline = line.replace('pretrained_embedding_file', embedding_file)
                        fout.write(newline)
                    else:
                        fout.write(line)
            fout.close()
