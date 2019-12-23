import sys

sys.path.append('../')

import argparse
import os
from collections import defaultdict

from data_util import io_helper

lowercase = True


def write_vocab(vocab, target_dir):
    io_helper.ensure_dir(target_dir)
    # write separate files for every language
    for langId in vocab:
        target_file = os.path.join(target_dir, langId + '.txt')
        print('Writing to', target_file)
        fout = open(target_file, 'w')
        # write every sentence and their indices (of words which are used in the probing tasks) to the file
        for sentence in sorted(vocab[langId].keys()):
            indices = list(sorted(set(vocab[langId][sentence])))
            for i in range(0, len(indices)):
                indices[i] = str(indices[i])
            fout.write(sentence + "\t" + " ".join(indices) + '\n')
        fout.close()


def read_probing(prob_file, vocab, langId):
    """"
    Reads the probing task file and add it to the vocab.
    :param prob_file: Path of probing task file
    :param vocab: dict to which the vocab will be added
    :param langId: id of the language of the probing file
    :return: dict with language ids as key and (dict with sentences (str) as keys and the indices of the sentence as values) as values
    """
    with open(prob_file) as f:
        for line in f:
            # Extract the sentence and the index of every linw
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            v = line.split('\t')
            sentence = v[0]
            if lowercase:
                sentence = sentence.lower()
            # Save sentence and index
            vocab[langId][sentence].append(int(v[1]))
    return vocab


def main(path, target_dir):
    languages = ['arabic', 'bulgarian', 'chinese', 'english', 'french', 'hindi', 'urdu', 'vietnamese', 'turkish',
                 'german', 'russian', 'spanish', 'finnish']
    vocab = defaultdict(lambda: defaultdict(list))
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
    parser = argparse.ArgumentParser()

    # argument options
    parser.add_argument('--output', type=str, required=False, default='../intrinsic/sentences',
                        help='output directory for the vocabulary files')
    parser.add_argument('--input', type=str, required=False, default='../intrinsic/data_contextual',
                        help='diretory of the dataset')

    args = parser.parse_args()
    main(args.input, args.output)
