# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-02-06 17:56:15
# @Last Modified by:   claravania
# @Last Modified time: 2019-02-07 11:04:26


import os
import sys
import codecs

from collections import defaultdict

lowercase=True

def write_vocab(vocab, target_dir):
	for langId in vocab:
		target_file = os.path.join(target_dir, langId + '.txt')
		print('Writing to', target_file)
		fout = open(target_file, 'w')
		for word in sorted(vocab[langId].keys()):
			fout.write(word + '\n')
		fout.close()


def read_xnli(xnli_file, vocab, langId):
	with open(xnli_file) as f:
		line_count = 0
		for line in f:
			if line_count == 0:
				line_count += 1
				continue
			sent1, sent2, _ = line.strip().split('\t')
			all_sents = sent1 + ' ' + sent2
			for word in all_sents.split():
				if lowercase:
					word = word.lower()
				vocab[langId][word] += 1
			line_count += 1
	return vocab


def read_ud(ud_file, vocab, langId):
	with open(ud_file) as f:
		for line in f:
			line = line.strip()
			if line == '' or line.startswith('#'):
				continue
			tokens = line.split('\t')
			if '.' in tokens[0] or '-' in tokens[0]:
				continue
			word = tokens[1]
			if lowercase:
				word = word.lower()
			vocab[langId][word] += 1
	return vocab


def read_conllu(conllu_file, vocab, langId):
	with open(conllu_file) as f:
		for line in f:
			line = line.strip()
			if line == '' or line.startswith('#'):
				continue
			tokens = line.split('\t')
			word = tokens[1]
			if lowercase:
				word = word.lower()
			vocab[langId][word] += 1
	return vocab


def read_conll(conll_file, vocab, langId):
	with open(conll_file) as f:
		for line in f:
			line = line.strip()
			if line == '' or line.startswith('#'):
				continue
			tokens = line.split()
			word = tokens[0]
			if lowercase:
				word = word.lower()
			vocab[langId][word] += 1
	return vocab


def main():
	languages = ['tr', 'de', 'ru', 'es', 'fi']
	target_dir = 'words/extrinsic_lower'
	vocab = defaultdict(lambda: defaultdict(int))

	xnli_dir = '../downstream_multilingual_data/xnli'
	for langId in languages:
		xnli_file = os.path.join(xnli_dir, 'multinli.train.' + langId + '.tsv')
		if not os.path.isfile(xnli_file):
			continue
		else:
			vocab = read_xnli(xnli_file, vocab, langId)


	ud_dir = '../downstream_multilingual_data/ud'
	ud_tb = {'fi': 'UD_Finnish-TDT', 'tr': 'UD_Turkish-IMST', 'es': 'UD_Spanish-AnCora', 'ru': 'UD_Russian-SynTagRus', 'de': 'UD_German-GSD'}
	for langId in ud_tb:
		ud_tb_dir = os.path.join(ud_dir, ud_tb[langId])
		filenames = os.listdir(ud_tb_dir)
		for filename in filenames:
			if filename.endswith('.conllu'):
				vocab = read_ud(os.path.join(ud_tb_dir, filename), vocab, langId)


	srl_dir = '../downstream_multilingual_data/srl'
	srl_lang = {'fi': 'CoNLL-UD-Finnish', 'tr': 'CoNLL2009-ST-Turkish', 'es': 'CoNLL2009-ST-Spanish', 'de': 'CoNLL2009-ST-German'}
	for langId in srl_lang:
		srl_lang_dir = os.path.join(srl_dir, srl_lang[langId])
		filenames = os.listdir(srl_lang_dir)
		if langId == 'fi':
			for filename in filenames:
				if filename.endswith('.conllu'):
					vocab = read_conllu(os.path.join(srl_lang_dir, filename), vocab, langId)
		else:
			for filename in filenames:
				vocab = read_conllu(os.path.join(srl_lang_dir, filename), vocab, langId)


	ner_dir = '../downstream_multilingual_data/ner'
	ner_lang = ['tr', 'de', 'ru', 'es', 'fi']
	for langId in ner_lang:
		ner_lang_dir = os.path.join(ner_dir, langId)
		filenames = os.listdir(ner_lang_dir)
		for filename in filenames:
			vocab = read_conll(os.path.join(ner_lang_dir, filename), vocab, langId)


	write_vocab(vocab, target_dir)




if __name__ == '__main__':
	main()