# -*- coding: utf-8 -*-

import os
import codecs
import json

from collections import defaultdict


languages = ['es', 'de', 'ru', 'tr']
# languages = ['de']


vocab = {}
dev_data = {}
test_data = {}

for lang in languages:
	dev_data[lang] = []
	test_data[lang] = []
	vocab[lang] = defaultdict(int)


def add_to_vocab(sent1, sent2, vocab):
	for w in sent1.split():
		vocab[w] += 1
	for w in sent2.split():
		vocab[w] += 1
	return vocab


# ################ READ TRAINING DATA ################

data_dir = "data/XNLI-MT-1.0/XNLI-MT-1.0/multinli"
labels = set()

data = []
for lang in languages:
	file_path = os.path.join(data_dir, "multinli.train." + lang + ".tsv")
	target_file = os.path.join(data_dir, "multinli.train." + lang + ".jsonl")
	with codecs.open(file_path, encoding="utf-8") as f, codecs.open(target_file, "w", encoding="utf-8") as out:
		count = 0
		for line in f:
			premise, hypo, label = line.strip().lower().split('\t')
			data = {"sentence1": premise, "sentence2": hypo, "gold_label": label}
			labels.add(label)
			vocab[lang] = add_to_vocab(premise, hypo, vocab[lang])
			if count > 0:
				json.dump(data, out, ensure_ascii=False)
				out.write("\n")
			count += 1

print(labels)


################ READ DEV/TEST DATA ################

dev_file = "data/XNLI-1.0/XNLI-1.0/xnli.dev.tsv"
test_file = "data/XNLI-1.0/XNLI-1.0/xnli.test.tsv"



with codecs.open(dev_file, "r", encoding="utf-8") as f:
	for line in f:
		items = line.strip().lower().split('\t')
		lang = items[0]
		if lang in languages:
			premise = items[16]
			hypo = items[17]
			label = items[1]
			dev_data[lang].append({"sentence1": premise, "sentence2": hypo, "gold_label": label})
			vocab[lang] = add_to_vocab(premise, hypo, vocab[lang])


for lang in languages:
	json_file = codecs.open(dev_file.replace('tsv', lang + '.jsonl'), "w", encoding="utf-8")
	count = 0
	for data in dev_data[lang]:
		json.dump(data, json_file, ensure_ascii=False)
		json_file.write("\n")
		count += 1
	json_file.close()
	print('Dev', lang, ":", count)



with codecs.open(test_file, "r", encoding="utf-8") as f:
	for line in f:
		items = line.strip().lower().split('\t')
		lang = items[0]
		if lang in languages:
			premise = items[16]
			hypo = items[17]
			label = items[1]
			test_data[lang].append({"sentence1": premise, "sentence2": hypo, "gold_label": label})
			vocab[lang] = add_to_vocab(premise, hypo, vocab[lang])


for lang in languages:
	json_file = codecs.open(test_file.replace('tsv', lang + '.jsonl'), "w", encoding="utf-8")
	count = 0
	for data in test_data[lang]:
		json.dump(data, json_file, ensure_ascii=False)
		json_file.write("\n")
		count += 1
	json_file.close()
	print('Test', lang, ":", count)



# write vocab to a file

for lang in languages:
	vocab_file = codecs.open("data/vocab/" + lang + '.txt', 'w', encoding="utf=8")
	for w in vocab[lang]:
		vocab_file.write(w + '\n')
	vocab_file.close()

