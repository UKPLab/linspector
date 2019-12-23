import sys

sys.path.append('../')

import os
import pickle
from collections import defaultdict

from data_util.io_helper import ensure_dir_for_file


def read_file(file, label_list):
    with open(file) as f:
        for line in f:
            line = line.strip()
            label = line.split("\t")[2]
            if label not in label_list:
                label_list.append(label)


def gen_pickle_file():
    # Load data

    test_vs_lang = defaultdict(dict)

    path = '../intrinsic/data_contextual/'

    tests = os.listdir(path)

    # Count number of tokens of a feature per language
    for test in tests:
        if not os.path.isdir(path + test):
            continue
        langs = os.listdir(path + test)

        for lang in langs:
            labels = []
            prefix = path + test + "/" + lang + "/"
            read_file(prefix + "train.txt", labels)
            read_file(prefix + "dev.txt", labels)
            read_file(prefix + "test.txt", labels)

            test_vs_lang[test][lang] = labels

    # Write to file
    print(test_vs_lang)
    pickle_file = "../intrinsic/data_contextual/features.pkl"
    ensure_dir_for_file(pickle_file)
    pickle.dump(test_vs_lang, open(pickle_file, "wb"))


if __name__ == "__main__":
    gen_pickle_file()
