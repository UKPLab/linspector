# -*- coding: utf-8 -*-
import sys

sys.path.append('../')

from data_util import generate_contextual_features_pickle

import math

from data_util.io_helper import ensure_dir_for_file
from data_util.schema import *
from data_util.reader import *

import random
import argparse


def get_supported_features_dict(threshold):
    """
    Returns dictionary which contains the languages and their supported tasks
    :param threshold: Number of instances which have to be available for the task
    :return: Dictionary with languages as keys and a list of supported features
    """
    # Load data
    data = load_ds("sigmorphon19-2")
    schema = UnimorphSchema()
    lang_vs_test = defaultdict(list)

    # Count number of tokens of a feature per language
    for lang in data:
        lang_data = data[lang]
        features = dict()
        for sentence in lang_data:
            for token in sentence:
                res = schema.decode_msd(token["msd"])
                x_feats = res[0]
                for feat in x_feats:
                    if feat in features:
                        features[feat] = features[feat] + 1
                    else:
                        features[feat] = 1

        # Add language to list if a feature has more than 10K tokens which support this feature
        for feat in features:
            if features[feat] >= threshold:
                lang_vs_test[lang].append(feat)
    return lang_vs_test


def reverse_dict_list(orig_dict):
    rev_dict = dict()
    for test in orig_dict:
        for lang in orig_dict[test]:
            if lang not in rev_dict:
                rev_dict[lang] = [test]
            else:
                rev_dict[lang].append(test)
    return rev_dict


def split_for_tests(data, feat, lang, savedir, threshold, z_score):
    """
    Splits unimorph data into training, dev an test for the given feature and language.
    :param data:
    :param feat: Case, Valency...
    :param lang: turkish, russian, english...
    :param savedir: folder to save the splits
    :param threshold: fixed to 10K
    :param z_score: Sentences which length doesn't match the z-score will be ignored
    """

    instances = []

    # Load data
    schema = UnimorphSchema()

    # Get length distribution
    lengths = defaultdict(int)
    for sentence in data[lang]:
        length = len(sentence)
        lengths[length] += 1
    # Get statistics
    expectation_value = 0
    number_of_sentences = 0
    for length in lengths:
        sentences_with_this_length = lengths[length]
        number_of_sentences += sentences_with_this_length
        expectation_value += sentences_with_this_length * length
    expectation_value = expectation_value / number_of_sentences
    variance = 0
    for length in lengths:
        sentences_with_this_length = lengths[length]
        variance += sentences_with_this_length * math.pow(length - expectation_value, 2)
    variance = variance / number_of_sentences
    std = math.sqrt(variance)
    min_length = expectation_value - z_score * std
    max_length = expectation_value + z_score * std

    for sentence in data[lang]:

        # Write whole sentence in a list
        forms = []
        for x in sentence:
            forms.append(x["form"])

        # Check length
        if len(forms) < min_length or len(forms) > max_length:
            continue

        word_index = -1
        for x in sentence:
            word_index += 1
            # exclude forms with space
            if ' ' in x["form"]:
                continue

            x_feats = schema.decode_msd(x["msd"])[0]

            if feat in x_feats:

                # Exceptions: drop V.PTCP from the case and gender tests - russian
                if (feat in ['Case', 'Gender']) and (lang == 'russian') and 'Part of Speech' in x_feats and (
                        x_feats['Part of Speech'] == 'Participle'):
                    continue

                # Exceptions: If it is a gender test and the noun does not have a gender feature, ignore
                if (feat == 'Gender') and (lang == 'russian') and 'Part of Speech' in x_feats and (
                        x_feats['Part of Speech'] == 'Noun') and (
                        'Gender' not in x_feats):
                    continue

                # Feature Person should have information about number too. (e.g. first person singular)
                if feat == 'Person':
                    if 'Number' not in x_feats:
                        continue
                    x_feats['Person'] = x_feats['Person'] + " " + x_feats['Number']

                y = {"forms": forms, "word_index": word_index, "feat": x_feats[feat], "msd": x["msd"][0]}
                instances.append(y)


            elif feat == "TagCount":
                num_morph_tags = str(len(x_feats))
                y = {"forms": forms, "word_index": word_index, "feat": num_morph_tags, "msd": ""}
                instances.append(y)
            elif feat == "CharacterBin":
                num_chars = len(x["form"])
                bins = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 100: 6}
                len_bin = 0
                for max_length in sorted(bins.keys()):
                    if num_chars <= max_length and len_bin == 0:
                        len_bin = bins[max_length]
                y = {"forms": forms, "word_index": word_index, "feat": str(len_bin), "msd": ""}
                instances.append(y)
            elif feat == "CharacterBin2":
                num_chars = len(x["form"])
                bins = {4: 1, 8: 2, 12: 3, 16: 4, 20: 5, 100: 6}
                len_bin = 0
                for max_length in sorted(bins.keys()):
                    if num_chars <= max_length and len_bin == 0:
                        len_bin = bins[max_length]
                y = {"forms": forms, "word_index": word_index, "feat": str(len_bin), "msd": ""}
                instances.append(y)

    if threshold > len(instances):
        print("Not enough instances left " + str(len(instances)))
        return

    # Random sample of all instances
    shuffled_instances = random.sample(instances, threshold)

    # Create three sets of this sample for training, dev and test
    train_inst = shuffled_instances[:int(threshold * 0.7)]
    dev_inst = shuffled_instances[int(threshold * 0.7):int(threshold * 0.9)]
    test_inst = shuffled_instances[int(threshold * 0.9):]

    train_path = os.path.join(savedir, feat, lang, "train.txt")
    ensure_dir_for_file(train_path)
    dev_path = os.path.join(savedir, feat, lang, "dev.txt")
    ensure_dir_for_file(dev_path)
    test_path = os.path.join(savedir, feat, lang, "test.txt")
    ensure_dir_for_file(test_path)

    # Write file
    write_to_file(train_inst, train_path)
    write_to_file(dev_inst, dev_path)
    write_to_file(test_inst, test_path)


def write_to_file(instances, file_path):
    """
    Writes the instances to a file. This will create a test file with the format:
     sentence word_index feature
    :param instances: list of instances
    :param file_path: path of the file
    """
    with open(file_path, 'w') as fout:
        for inst in instances:
            fout.write(" ".join(inst["forms"]) + "\t" + str(inst["word_index"]) + "\t" + inst["feat"] + "\n")

    fout.close()


def main(args):
    lang_vs_test = get_supported_features_dict(args.size)

    for lang in lang_vs_test:
        if args.lang is None or lang.lower() in args.lang.lower().split(","):
            data = load_ds("sigmorphon19-2", lang)
            # Create tests for every feature of this language which contains more than 10K tokens
            for test_name in lang_vs_test[lang]:
                if args.feat is None or test_name.lower() in args.feat.lower().split(","):
                    print("Preparing " + lang + " - " + test_name)
                    split_for_tests(data, test_name, lang, args.savedir, args.size, args.zscore)
            if args.feat is None or "tagcount" in args.feat.lower().split(","):
                print("Preparing " + lang + " - TagCount")
                split_for_tests(data, "TagCount", lang, args.savedir, args.size, args.zscore)
            if args.feat is None or "characterbin" in args.feat.lower().split(","):
                print("Preparing " + lang + " - CharacterBin")
                split_for_tests(data, "CharacterBin", lang, args.savedir, args.size, args.zscore)
            if args.feat is None or "characterbin2" in args.feat.lower().split(","):
                print("Preparing " + lang + " - CharacterBin2")
                split_for_tests(data, "CharacterBin2", lang, args.savedir, args.size, args.zscore)
    # Generate pickle which contains which language supports which feature and which labels they have
    generate_contextual_features_pickle.gen_pickle_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Prepare feature tests
    parser.add_argument('--savedir', type=str, default='../intrinsic/data_contextual')
    parser.add_argument('--zscore', type=float, default=2, help="sentence lengths")
    parser.add_argument('--lang', type=str, default=None,
                        help="The languages. Multiple languages can be separated by a comma")
    parser.add_argument('--feat', type=str, default=None,
                        help="The features. Multiple features can be separated by commas")
    parser.add_argument('--size', type=int, default=10000, help="The size of the dataset for one task")

    args = parser.parse_args()

    main(args)
