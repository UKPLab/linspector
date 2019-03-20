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


def read_probing(prob_file, vocab, langId):
    with open(prob_file) as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            tokens = line.split('\t')
            word = tokens[0]
            if lowercase:
                word=word.lower()
            vocab[langId][word] += 1
    return vocab


def read_pair_probing(pair_prob_file, vocab, langId):
    with open(pair_prob_file) as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            tokens = line.split('\t')
            word = tokens[0]
            if lowercase:
                word=word.lower()
            vocab[langId][word] += 1
            word = tokens[1]
            if lowercase:
                word=word.lower()
            vocab[langId][word] += 1
    return vocab



def main():
    languages = ['turkish', 'german', 'russian', 'spanish', 'finnish']
    target_dir = 'words/intrinsic_lower'
    path = '../probing_datasets'

    vocab = defaultdict(lambda: defaultdict(int))
    tests = ['Polarity', 'Tense', 'Case', 'Mood', 'POS', 'CharacterBin', 'Pseudo', 'Possession', 'Voice', 'Person', 'TagCount', 'Gender', 'Number']
    pairtests = ['OddFeat', 'SameFeat']

    for test in tests:
        for lang in languages:
            train_file = os.path.join(path, test, lang, 'train.txt')
            dev_file = os.path.join(path, test, lang, 'dev.txt')
            test_file = os.path.join(path, test, lang, 'test.txt')
            if not os.path.isfile(train_file):
                continue
            else:
                vocab = read_probing(train_file, vocab, lang)
                vocab = read_probing(dev_file, vocab, lang)
                vocab = read_probing(test_file, vocab, lang)

    for test in pairtests:
        for lang in languages:
            train_file = os.path.join(path, test, lang, 'train.txt')
            dev_file = os.path.join(path, test, lang, 'dev.txt')
            test_file = os.path.join(path, test, lang, 'test.txt')
            if not os.path.isfile(train_file):
                continue
            else:
                vocab = read_pair_probing(train_file, vocab, lang)
                vocab = read_pair_probing(dev_file, vocab, lang)
                vocab = read_pair_probing(test_file, vocab, lang)

    write_vocab(vocab, target_dir)


if __name__ == '__main__':
    main()
