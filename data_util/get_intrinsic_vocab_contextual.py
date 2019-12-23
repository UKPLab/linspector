import sys

sys.path.append('../')

import os
from collections import defaultdict

from data_util.io_helper import ensure_dir

lowercase = True


def write_vocab(vocab, target_dir):
    ensure_dir(target_dir)
    # write separate files for every language
    for langId in vocab:
        target_file = os.path.join(target_dir, langId + '.txt')
        print('Writing to', target_file)
        fout = open(target_file, 'w')
        # Write every word into the file
        for word in sorted(vocab[langId].keys()):
            fout.write(word + '\n')
        fout.close()


def read_probing(prob_file, vocab, langId):
    """"
        Reads the probing task file and add it to the vocab.
        :param prob_file: Path of probing task file
        :param vocab: dict to which the vocab will be added
        :param langId: id of the language of the probing file
        :return: dict with language ids as key and (dict with words (str) as keys and the number of appearances as values) as values
        """
    with open(prob_file) as f:
        for line in f:
            # Extract the relevant word of every line
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            v = line.split('\t')
            sentence = v[0]
            word_index = int(v[1])
            words = sentence.split(" ")
            word = words[word_index]
            if lowercase:
                word = word.lower()
            # Save the word
            vocab[langId][word] += 1
    return vocab


def main():
    languages = ['turkish', 'german', 'russian', 'spanish', 'finnish']
    target_dir = '../intrinsic/words/'
    path = '../intrinsic/data_contextual'

    vocab = defaultdict(lambda: defaultdict(int))
    tests = os.listdir(path)

    # Read train, dev, test files of every probing task
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

    # Save vocab
    write_vocab(vocab, target_dir)


if __name__ == '__main__':
    main()
