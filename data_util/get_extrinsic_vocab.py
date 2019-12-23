# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-02-06 17:56:15
# @Last Modified by:   claravania
# @Last Modified time: 2019-02-07 11:04:26
import sys

sys.path.append('../')


import os
from collections import defaultdict

from data_util import io_helper

lowercase = True


def write_vocab(vocab, target_dir):
    io_helper.ensure_dir(target_dir)
    for langId in vocab:
        target_file = os.path.join(target_dir, langId + '.txt')
        print('Writing to', target_file)
        fout = open(target_file, 'w')
        for word in sorted(vocab[langId].keys()):
            fout.write(word + '\n')
        fout.close()


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


def read_pos(file, vocab, lang):
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            v = line.split(" ")
            for word_pos in v:
                if "###" in word_pos:
                    word = "#"
                else:
                    word = word_pos.split("##")[-2]

                if lowercase:
                    word = word.lower()
                vocab[lang][word] += 1
    return vocab


def copy_vocab_file(vocab_file, vocab, langId):
    with open(vocab_file) as f:
        for word in f:
            word = word.strip()
            vocab[langId][word] += 1
    return vocab


def main():
    languages = ['finnish', 'german', 'russian', 'spanish', 'turkish']
    target_dir = '../extrinsic/words/'
    vocab = defaultdict(lambda: defaultdict(int))

    pos_dir = '../extrinsic/data/ud_pos'
    for lang in languages:
        pos = os.path.join(pos_dir, lang)
        files = os.listdir(pos)
        for file in files:
            if file.endswith(".pos"):
                print("POS reading " + file)
                vocab = read_pos(os.path.join(pos, file), vocab, lang)

    xnli_dir = '../extrinsic/data/xnli/vocab'
    # We have already a vocab file for XNLI through preprocessing. Copy that vocabulary
    for langId in languages:
        vocab_file = os.path.join(xnli_dir, langId + ".txt")
        if os.path.exists(vocab_file):
            vocab = copy_vocab_file(vocab_file, vocab, langId)
        else:
            print("XNLI: File does not exist " + vocab_file)

    ud_dir = '../extrinsic/data/ud_dep'
    for langId in languages:
        ud_tb_dir = os.path.join(ud_dir, langId)
        filenames = os.listdir(ud_tb_dir)
        for filename in filenames:
            if filename.endswith('.conllu'):
                print("UD Reading: " + filename)
                vocab = read_ud(os.path.join(ud_tb_dir, filename), vocab, langId)

    srl_dir = '../extrinsic/data/srl'
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

    ner_dir = '../extrinsic/data/ner'
    for langId in languages:
        ner_lang_dir = os.path.join(ner_dir, langId)
        filenames = os.listdir(ner_lang_dir)
        for filename in filenames:
            print("NER Reading " + filename)
            vocab = read_conll(os.path.join(ner_lang_dir, filename), vocab, langId)

    write_vocab(vocab, target_dir)


if __name__ == '__main__':
    main()
