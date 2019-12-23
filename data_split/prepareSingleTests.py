# -*- coding: utf-8 -*-
import sys

sys.path.append('../')

import argparse
import pickle
import random

from data_split.util import *
from data_util.reader import *
from data_util.schema import *


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def reverse_dict_list(orig_dict):
    rev_dict = dict()
    for test in orig_dict:
        for lang in orig_dict[test]:
            if lang not in rev_dict:
                rev_dict[lang] = [test]
            else:
                rev_dict[lang].append(test)
    return rev_dict


def get_mixed_surface(feat, lang, vocab, threshold):
    """
    Get cnt number of frequent surface and lemma which DOES not contain the 'feat'
    None of the surface forms should have the feat
    :param feat: morphological feature
    :param lang: language
    :param vocab: list of frequent words
    :param cnt: number of desired nonsense labels
    :return: list of word, label tuple
    """
    freq_surf = []
    rare_surf = []

    schema = UnimorphSchema()
    data = load_ds("unimorph", lang)
    forbid_vocab = dict()

    # make a vocabulary of forbidden words
    for x in data[lang]:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue
        x_feats = schema.decode_msd(x["msd"])[0]
        if feat in x_feats:
            forbid_vocab[x["form"]] = 1

    # include each surface form once
    surf_cnt = dict()

    for x in data[lang]:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue

        # if any of the surface forms have the feat, exclude them
        if x["form"] in forbid_vocab:
            continue

        # exclude x, if the surface form is already in the data
        if x["form"] in surf_cnt:
            continue
        else:
            surf_cnt[x['form']] = 1

        x_feats = schema.decode_msd(x["msd"])[0]

        ## Exceptions: drop V.PTCP from the case and gender tests - russian
        if (feat in ['Case', 'Gender']) and (lang == 'russian') and (x_feats['Part of Speech'] == 'Participle'):
            continue

        ## Exceptions: If it is a gender test and the noun does not have a gender feature, ignore
        if (feat == 'Gender') and (lang == 'russian') and (x_feats['Part of Speech'] == 'Noun') and (
                'Gender' not in x_feats):
            continue

        if feat not in x_feats:
            # flag x
            x["flag"] = 1
            if x["form"].lower() in vocab:
                freq_surf.append(x)
            else:
                rare_surf.append(x)
        """
        else:
            if feat == 'Number' and (x_feats['Part of Speech'] != 'Noun') and (feat in x_feats):
                x["flag"] = 1
                if x["form"].lower() in vocab:
                    freq_surf.append(x)
                else:
                    rare_surf.append(x)
        """
    # Try to sample 80%-20% if possible
    if (len(freq_surf) >= int(threshold * 0.8)) and (len(rare_surf) >= int(threshold * 0.2)):
        shuffled_frequent = random.sample(freq_surf, int(threshold * 0.8))
        shuffled_rare = random.sample(rare_surf, int(threshold * 0.2))
        instances = shuffled_frequent + shuffled_rare
    # else get all the frequent ones, and sample the rest from the rare ones
    elif (len(freq_surf) + len(rare_surf)) >= threshold:
        shuffled_frequent = random.sample(freq_surf, len(freq_surf))
        shuffled_rare = random.sample(rare_surf, int(threshold - len(freq_surf)))
        instances = shuffled_frequent + shuffled_rare
    else:
        print("Not enough instances are left")
        return []

    return instances


def split_for_morph_test_mixed(feat, lang, vocab, nonlabelratio, savedir, threshold=10000):
    """
    Splits unimorph data into training, dev an test for the given feature and language.
    Precheck 1: We check if the feature can have more than one label beforehand
    This function eliminates cases where the form has space - e.g., "anlıyor musun"
    This function eliminates cases where the feature is very sparse (seen less than 5 times)
    This function eliminates ambiguous forms
    :param feat: Case, Valency...
    :param lang: turkish, russian, english...
    :param vocab: frequent word list from wikipedia
    :param savedir: folder to save the splits
    :param threshold: fixed to 10K
    :return: Default output directory is ./output/feature/lang/train-dev-test.txt
    """

    freq_surf = []
    rare_surf = []

    schema = UnimorphSchema()
    data = load_ds("unimorph", lang)

    # make a label dictionary for noisy labels
    label_cnt = dict()
    # make a surface form dictionary for ambiguous
    surf_cnt = dict()

    nonlabel_cnt = threshold * nonlabelratio
    reallabel_cnt = threshold * (1. - nonlabelratio)

    for x in data[lang]:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue

        x_feats = schema.decode_msd(x["msd"])[0]
        if feat in x_feats:
            # There is a bug with Number: 'Part of Speech'
            # Don't include verbs/verb like words to singular/plural test
            # if feat=='Number' and (x_feats['Part of Speech']!='Noun'):
            #    continue

            ## Exceptions: drop V.PTCP from the case and gender tests - russian
            if (feat in ['Case', 'Gender']) and (lang == 'russian') and (x_feats['Part of Speech'] == 'Participle'):
                continue

            ## Exceptions: If it is a gender test and the noun does not have a gender feature, ignore
            if (feat == 'Gender') and (lang == 'russian') and (x_feats['Part of Speech'] == 'Noun') and (
                    'Gender' not in x_feats):
                continue

            if x["form"].lower() in vocab:
                freq_surf.append(x)
            # rare surface and frequent lemma
            else:
                rare_surf.append(x)

            # for sparse labels
            if x_feats[feat] not in label_cnt:
                label_cnt[x_feats[feat]] = 1
            else:
                label_cnt[x_feats[feat]] += 1
            # for amb. forms
            if x['form'] not in surf_cnt:
                surf_cnt[x['form']] = 1
            else:
                surf_cnt[x['form']] += 1

    # if there is any (very) sparse label, exclude those
    forbid_labs = []
    for label in label_cnt:
        if (label_cnt[label]) < 5:
            forbid_labs.append(label)

    # if there are any surface forms with multiple values, exclude those
    amb_form_dict = dict()
    for surf, cnt in surf_cnt.items():
        if cnt > 1:
            amb_form_dict[surf] = 1

    # check here if we don't have enough instances or labels already
    if ((len(label_cnt) - len(forbid_labs)) < 2) or len(surf_cnt) < reallabel_cnt:
        print("Not enough instances or labels are left")
        return False

    # Exclude the noisy labels, ambiguous forms and rare words
    if ((len(forbid_labs) > 0) or (len(amb_form_dict) > 0)):
        freq_surf = []
        rare_surf = []

        for x in data[lang]:
            # exclude lemmas with space
            if ' ' in x["form"]:
                continue
            # exclude amb. forms
            if x["form"] in amb_form_dict:
                continue

            x_feats = schema.decode_msd(x["msd"])[0]

            if 'Part of Speech' not in x_feats:
                # probably a mistake in unimorph, just pass
                continue

            # exclude non nominal forms which has plurality tag
            # if feat=='Number' and (x_feats['Part of Speech']!='Noun'):
            #    continue
            ## Exceptions: drop V.PTCP from the case and gender tests - russian
            if (feat in ['Case', 'Gender']) and (lang == 'russian') and x_feats['Part of Speech'] == 'Participle':
                continue

            ## Exceptions: If it is a gender test and the noun does not have a gender feature, ignore
            if (feat == 'Gender') and (lang == 'russian') and (x_feats['Part of Speech'] == 'Noun') and (
                    'Gender' not in x_feats):
                continue

            if (feat in x_feats) and (x_feats[feat] not in forbid_labs):
                # instances.append(x)
                # if frequent surface
                if x["form"].lower() in vocab:
                    freq_surf.append(x)
                # rare surface
                else:
                    rare_surf.append(x)

    # Try to sample 80%-20% if possible
    if (len(freq_surf) >= int(reallabel_cnt * 0.8)) and (len(rare_surf) >= int(reallabel_cnt * 0.2)):
        shuffled_frequent = random.sample(freq_surf, int(reallabel_cnt * 0.8))
        shuffled_rare = random.sample(rare_surf, int(reallabel_cnt * 0.2))
        instances = shuffled_frequent + shuffled_rare
    # else get all the frequent ones, and sample the rest from the rare ones
    elif (len(freq_surf) + len(rare_surf)) >= reallabel_cnt:
        shuffled_frequent = random.sample(freq_surf, len(freq_surf))
        shuffled_rare = random.sample(rare_surf, int(reallabel_cnt - len(freq_surf)))
        instances = shuffled_frequent + shuffled_rare
    else:
        print("Not enough instances are left")
        return False

    # get the nonlabel instances
    non_instances = get_mixed_surface(feat, lang, vocab, nonlabel_cnt)

    if len(non_instances) == 0:
        return False

    all_instances = instances + non_instances
    shuffled_instances = random.sample(all_instances, threshold)

    train_inst = shuffled_instances[:int(threshold * 0.7)]
    dev_inst = shuffled_instances[int(threshold * 0.7):int(threshold * 0.9)]
    test_inst = shuffled_instances[int(threshold * 0.9):]

    train_path = os.path.join(savedir, feat, lang, "train.txt")
    ensure_dir(train_path)
    dev_path = os.path.join(savedir, feat, lang, "dev.txt")
    ensure_dir(dev_path)
    test_path = os.path.join(savedir, feat, lang, "test.txt")
    ensure_dir(test_path)

    # Write file
    with open(train_path, 'w') as fout:
        for inst in train_inst:
            x_feats = schema.decode_msd(inst["msd"])[0]
            if "flag" not in inst:
                if feat == 'Person':
                    x_feats[feat] = x_feats[feat] + " " + x_feats['Number']
                fout.write("\t".join([inst["form"], x_feats[feat]]) + "\n")
            else:
                fout.write("\t".join([inst["form"], "None"]) + "\n")
    fout.close()

    with open(dev_path, 'w') as fout:
        for inst in dev_inst:
            x_feats = schema.decode_msd(inst["msd"])[0]
            if "flag" not in inst:
                if feat == 'Person':
                    x_feats[feat] = x_feats[feat] + " " + x_feats['Number']
                fout.write("\t".join([inst["form"], x_feats[feat]]) + "\n")
            else:
                fout.write("\t".join([inst["form"], "None"]) + "\n")
    fout.close()

    with open(test_path, 'w') as fout:
        for inst in test_inst:
            x_feats = schema.decode_msd(inst["msd"])[0]
            if "flag" not in inst:
                if feat == 'Person':
                    x_feats[feat] = x_feats[feat] + " " + x_feats['Number']
                fout.write("\t".join([inst["form"], x_feats[feat]]) + "\n")
            else:
                fout.write("\t".join([inst["form"], "None"]) + "\n")
    fout.close()

    return True


def split_for_morph_test(feat, lang, vocab, savedir, threshold=10000):
    """
    Splits unimorph data into training, dev an test for the given feature and language.
    Precheck 1: We check if the feature can have more than one label beforehand
    This function eliminates cases where the form has space - e.g., "anlıyor musun"
    This function eliminates cases where the feature is very sparse (seen less than 5 times)
    This function eliminates ambiguous forms
    :param feat: Case, Valency...
    :param lang: turkish, russian, english...
    :param vocab: frequent word list from wikipedia
    :param savedir: folder to save the splits
    :param threshold: fixed to 10K
    :return: Default output directory is ./output/feature/lang/train-dev-test.txt
    """

    freq_surf = []
    rare_surf = []

    schema = UnimorphSchema()
    data = load_ds("unimorph", lang)

    # make a label dictionary for noisy labels
    label_cnt = dict()
    # make a surface form dictionary for ambiguous
    surf_cnt = dict()

    for x in data[lang]:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue
        x_feats = schema.decode_msd(x["msd"])[0]
        if feat in x_feats:
            # There is a bug with Number: 'Part of Speech'
            # Don't include verbs/verb like words to singular/plural test
            # if feat=='Number' and (x_feats['Part of Speech']!='Noun'):
            #    continue

            # instances.append(x)
            if x["form"].lower() in vocab:
                freq_surf.append(x)
            # rare surface
            else:
                rare_surf.append(x)

            # for sparse labels
            if x_feats[feat] not in label_cnt:
                label_cnt[x_feats[feat]] = 1
            else:
                label_cnt[x_feats[feat]] += 1
            # for amb. forms
            if x['form'] not in surf_cnt:
                surf_cnt[x['form']] = 1
            else:
                surf_cnt[x['form']] += 1

    # if there is any (very) sparse label, exclude those
    forbid_labs = []
    for label in label_cnt:
        if (label_cnt[label]) < 5:
            forbid_labs.append(label)

    # if there are any surface forms with multiple values, exclude those
    amb_form_dict = dict()
    for surf, cnt in surf_cnt.items():
        if cnt > 1:
            amb_form_dict[surf] = 1

    # check here if we don't have enough instances or labels already
    if ((len(label_cnt) - len(forbid_labs)) < 2) or len(surf_cnt) < threshold:
        print("Not enough instances or labels are left")
        return False

    # Exclude the noisy labels, ambiguous forms and rare words
    if ((len(forbid_labs) > 0) or (len(amb_form_dict) > 0)):
        freq_surf = []
        rare_surf = []

        for x in data[lang]:
            # exclude lemmas with space
            if ' ' in x["form"]:
                continue
            # exclude amb. forms
            if x["form"] in amb_form_dict:
                continue

            x_feats = schema.decode_msd(x["msd"])[0]

            if 'Part of Speech' not in x_feats:
                # probably a mistake in unimorph, just pass
                continue

            # exclude non nominal forms which has plurality tag
            # if feat=='Number' and (x_feats['Part of Speech']!='Noun'):
            #    continue

            if (feat in x_feats) and (x_feats[feat] not in forbid_labs):
                # instances.append(x)
                # if frequent surface
                if x["form"].lower() in vocab:
                    freq_surf.append(x)
                # rare surface
                else:
                    rare_surf.append(x)

    # Try to sample 80%-20% if possible
    if (len(freq_surf) >= int(threshold * 0.8)) and (len(rare_surf) >= int(threshold * 0.2)):
        shuffled_frequent = random.sample(freq_surf, int(threshold * 0.8))
        shuffled_rare = random.sample(rare_surf, int(threshold * 0.2))
        instances = shuffled_frequent + shuffled_rare
    # else get all the frequent ones, and sample the rest from the rare ones
    elif (len(freq_surf) + len(rare_surf)) >= threshold:
        shuffled_frequent = random.sample(freq_surf, len(freq_surf))
        shuffled_rare = random.sample(rare_surf, int(threshold - len(freq_surf)))
        instances = shuffled_frequent + shuffled_rare
    else:
        print("Not enough instances are left")
        return False

    shuffled_instances = random.sample(instances, threshold)

    train_inst = shuffled_instances[:int(threshold * 0.7)]
    dev_inst = shuffled_instances[int(threshold * 0.7):int(threshold * 0.9)]
    test_inst = shuffled_instances[int(threshold * 0.9):]

    train_path = os.path.join(savedir, feat, lang, "train.txt")
    ensure_dir(train_path)
    dev_path = os.path.join(savedir, feat, lang, "dev.txt")
    ensure_dir(dev_path)
    test_path = os.path.join(savedir, feat, lang, "test.txt")
    ensure_dir(test_path)

    # Write file
    with open(train_path, 'w') as fout:
        for inst in train_inst:
            x_feats = schema.decode_msd(inst["msd"])[0]
            if feat == 'Person':
                x_feats[feat] = x_feats[feat] + " " + x_feats['Number']
            fout.write("\t".join([inst["form"], x_feats[feat]]) + "\n")
    fout.close()

    with open(dev_path, 'w') as fout:
        for inst in dev_inst:
            x_feats = schema.decode_msd(inst["msd"])[0]
            if feat == 'Person':
                x_feats[feat] = x_feats[feat] + " " + x_feats['Number']
            fout.write("\t".join([inst["form"], x_feats[feat]]) + "\n")
    fout.close()

    with open(test_path, 'w') as fout:
        for inst in test_inst:
            x_feats = schema.decode_msd(inst["msd"])[0]
            if feat == 'Person':
                x_feats[feat] = x_feats[feat] + " " + x_feats['Number']
            fout.write("\t".join([inst["form"], x_feats[feat]]) + "\n")
    fout.close()

    return True


def split_for_number_test(lang, vocab, savedir, threshold=10000):
    """
    Create train, dev, test splits for 'number of characters' and 'number of morphemes' tests
    :param lang: turkish, russian, english...
    :param vocab: frequent word list from wikipedia
    :param savedir: folder to save the splits
    :param threshold: fixed to 10K
    :return: Default output directory is ./output/CharacterCount/lang/train-dev-test.txt and
                                         ./output/TagCount/lang/train-dev-test.txt
    """
    # instances = []
    freq_surf = []
    rare_surf = []

    schema = UnimorphSchema()
    data = load_ds("unimorph", lang)
    surf_dict = dict()
    for x in data[lang]:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue
        # exclude duplicates
        if x['form'] in surf_dict:
            continue
        # exclude rare words
        # if x[raretype].lower() not in vocab:
        #    continue
        # else:
        surf_dict[x['form']] = 1
        x["num_chars"] = str(len(x["form"]))
        x["num_morph_tags"] = str(len(schema.decode_msd(x["msd"])[0]))

        if x["form"].lower() in vocab:
            freq_surf.append(x)
        # rare surface and frequent lemma
        else:
            rare_surf.append(x)
            # instances.append(x)

    # Try to sample 80%-20% if possible
    if (len(freq_surf) >= int(threshold * 0.8)) and (len(rare_surf) >= int(threshold * 0.2)):
        shuffled_frequent = random.sample(freq_surf, int(threshold * 0.8))
        shuffled_rare = random.sample(rare_surf, int(threshold * 0.2))
        instances = shuffled_frequent + shuffled_rare
    # else get all the frequent ones, and sample the rest from the rare ones
    elif (len(freq_surf) + len(rare_surf)) >= threshold:
        shuffled_frequent = random.sample(freq_surf, len(freq_surf))
        shuffled_rare = random.sample(rare_surf, int(threshold - len(freq_surf)))
        instances = shuffled_frequent + shuffled_rare
    else:
        print("Not enough instances are left")
        return False

    shuffled_instances = random.sample(instances, threshold)
    train_inst = shuffled_instances[:int(threshold * 0.7)]
    dev_inst = shuffled_instances[int(threshold * 0.7):int(threshold * 0.9)]
    test_inst = shuffled_instances[int(threshold * 0.9):]

    feat = "CharacterCount"
    train_path = os.path.join(savedir, feat, lang, "train.txt")
    ensure_dir(train_path)
    dev_path = os.path.join(savedir, feat, lang, "dev.txt")
    ensure_dir(dev_path)
    test_path = os.path.join(savedir, feat, lang, "test.txt")
    ensure_dir(test_path)

    # Write file
    with open(train_path, 'w') as fout:
        for inst in train_inst:
            fout.write("\t".join([inst["form"], inst["num_chars"]]) + "\n")
    fout.close()

    with open(dev_path, 'w') as fout:
        for inst in dev_inst:
            fout.write("\t".join([inst["form"], inst["num_chars"]]) + "\n")
    fout.close()

    with open(test_path, 'w') as fout:
        for inst in test_inst:
            fout.write("\t".join([inst["form"], inst["num_chars"]]) + "\n")
    fout.close()

    feat = "TagCount"
    train_path = os.path.join(savedir, feat, lang, "train.txt")
    ensure_dir(train_path)
    dev_path = os.path.join(savedir, feat, lang, "dev.txt")
    ensure_dir(dev_path)
    test_path = os.path.join(savedir, feat, lang, "test.txt")
    ensure_dir(test_path)

    # Write file
    with open(train_path, 'w') as fout:
        for inst in train_inst:
            fout.write("\t".join([inst["form"], inst["num_morph_tags"]]) + "\n")
    fout.close()

    with open(dev_path, 'w') as fout:
        for inst in dev_inst:
            fout.write("\t".join([inst["form"], inst["num_morph_tags"]]) + "\n")
    fout.close()

    with open(test_path, 'w') as fout:
        for inst in test_inst:
            fout.write("\t".join([inst["form"], inst["num_morph_tags"]]) + "\n")
    fout.close()
    return True


def split_for_nonsense(lang, pseudodir, savedir, type="ort", threshold=10000):
    """
    Create splits in two different formats:
    Binary: given the word, guess if it is pseduo or not
    Old20:  given the pseudo word, guess its level of nonsense - approximately
    Probably binary one makes more sense, but there are more options available
    :param lang: any supported-prcessed wuggy language under generated folder
    :param pseudodir: folder of pseudo files generated by wuggy
    :param savedir: folder to save the splits
    :param type: ort or phon
    :param threshold:
    :return:
    """
    instances = []
    words = []
    fin_path = os.path.join(pseudodir, (type + "_" + lang))

    # Read file
    i = 0
    with open(fin_path) as fin:
        for line in fin:
            if i == 0:
                i += 1
                continue
            x = {}
            all_cols = line.rstrip().split("\t")
            x["word"] = all_cols[0]
            words.append(x["word"])
            x["non_sense"] = all_cols[1]
            instances.append(x)
    fin.close()

    # make a vocab
    word_vocab = list(set(words))
    if len(instances) < threshold:
        print("Not enough instances")
        return False

    if len(word_vocab) < (threshold / 2):
        print("Not enough words")
        return False

    # shuffle is an in-place operation
    random.shuffle(word_vocab)
    shuffled_instances = random.sample(instances, threshold)
    shuffled_labels = np.random.choice([0, 1], size=(threshold,), p=[1. / 2, 1. / 2])

    train_inst = shuffled_instances[:int(threshold * 0.7)]
    train_labels = shuffled_labels[:int(threshold * 0.7)]
    dev_inst = shuffled_instances[int(threshold * 0.7):int(threshold * 0.9)]
    dev_labels = shuffled_labels[int(threshold * 0.7):int(threshold * 0.9)]
    test_inst = shuffled_instances[int(threshold * 0.9):]
    test_labels = shuffled_labels[int(threshold * 0.9):]

    feat = "NonSense_Binary"
    train_path = os.path.join(savedir, feat, lang, "train.txt")
    ensure_dir(train_path)
    dev_path = os.path.join(savedir, feat, lang, "dev.txt")
    ensure_dir(dev_path)
    test_path = os.path.join(savedir, feat, lang, "test.txt")
    ensure_dir(test_path)

    # Write file
    wi = 0
    with open(train_path, 'w') as fout:
        for inst, label in zip(train_inst, train_labels):
            if label == 0:
                fout.write("\t".join([inst["non_sense"], str(label)]) + "\n")
            elif label == 1:
                fout.write("\t".join([word_vocab[wi], str(label)]) + "\n")
                wi += 1
    fout.close()

    with open(dev_path, 'w') as fout:
        for inst, label in zip(dev_inst, dev_labels):
            if label == 0:
                fout.write("\t".join([inst["non_sense"], str(label)]) + "\n")
            elif label == 1:
                fout.write("\t".join([word_vocab[wi], str(label)]) + "\n")
                wi += 1
    fout.close()

    with open(test_path, 'w') as fout:
        for inst, label in zip(test_inst, test_labels):
            if label == 0:
                fout.write("\t".join([inst["non_sense"], str(label)]) + "\n")
            elif label == 1:
                fout.write("\t".join([word_vocab[wi], str(label)]) + "\n")
                wi += 1
    fout.close()

    return True


def main(args):
    langs = {'portuguese': 'pt',
             'french': 'fr',
             'serbo-croatian': 'sh',
             'polish': 'pl',
             'czech': 'cs',
             'modern-greek': 'el',
             'catalan': 'ca',
             'bulgarian': 'bg',
             'danish': 'da',
             'estonian': 'et',
             'quechua': 'qu',
             'swedish': 'sv',
             'armenian': 'hy',
             'macedonian': 'mk',
             'arabic': 'ar',
             'dutch': 'nl',
             'hungarian': 'hu',
             'italian': 'it',
             'romanian': 'ro',
             'ukranian': 'uk',
             'german': 'de',
             'finnish': 'fi',
             'russian': 'ru',
             'turkish': 'tr',
             'spanish': 'es'
             }

    # Language specific vocabulary sizes
    # wiki vocabulary sizes: de: 2275234, es: 985668, fi: 730484, tr: 416052, ru: 1888424,
    # pt: 592109, fr: 1152450, sh: 454675, pl: 1032578, cs:627842, el: 306450, ca:490566, bg: 334079, da: 312957
    # et: 329988, qu: 23074, sv: 1143274, hy: 332673, mk: 176948, ar: 610978, nl: 871023, hu: 793867, it: 871054,
    # ro: 354325, uk: 912459
    langs_vocab = {'german': 750000, 'finnish': 500000, 'russian': 750000, 'turkish': 500000, 'spanish': 500000, \
                   'portuguese': 500000, 'french': 750000, 'serbo-croatian': 500000, 'polish': 750000, 'czech': 500000, \
                   'modern-greek': 500000, 'catalan': 500000, 'bulgarian': 500000, 'danish': 500000, 'estonian': 500000, \
                   'quechua': 500000, 'swedish': 750000, 'armenian': 500000, 'macedonian': 500000, 'arabic': 500000, \
                   'dutch': 600000, 'hungarian': 600000, 'italian': 600000, 'romanian': 500000, 'ukranian': 750000}

    with open('test_vs_lang_feat_over_10K.pkl', 'rb') as handle:
        test_vs_lang = pickle.load(handle)
    lang_vs_test = reverse_dict_list(test_vs_lang)

    # Load preprocessed statistics
    with open('supported_languages_over_10K.pkl', 'rb') as handle:
        supported_lang_list = pickle.load(handle)

    if args.feat == 1:
        for lang in lang_vs_test:
            if lang in langs:
                embfile = os.path.join('..', "embeddings", "wiki." + langs[lang] + ".vec")
                print("Reading vocabulary for lang " + lang)
                vocab = load_dict(embfile, maxvoc=langs_vocab[lang])
                for test_name in lang_vs_test[lang]:
                    print("Preparing " + lang + " - " + test_name)
                    split_for_morph_test_mixed(test_name, lang, vocab, args.nonlabelratio, args.savedir)

    if args.common == 1:
        # General tests for all supported languages
        for lang in supported_lang_list:
            if lang in langs:
                # get the vocabulary first
                embfile = os.path.join('..', "embeddings", "wiki." + langs[lang] + ".vec")
                print("Reading vocabulary for lang " + lang)
                vocab = load_dict(embfile, maxvoc=langs_vocab[lang])
                print("Preparing Character and Tag Count Tests- " + lang)
                split_for_number_test(lang, vocab, args.savedir)
                print("Preparing POS Test- " + lang)
                split_for_morph_test("Part of Speech", lang, vocab, args.savedir)

    if args.pseudo == 1:
        # Pseudo word tests only for languages with wuggy support
        # Orthographic pseudo
        ort_lang_lst = ["turkish", "german", "spanish", "english", 'dutch', 'french', 'serbian_latin', 'basque',
                        'vietnamese']
        for lang in ort_lang_lst:
            print("Processing orthographic " + lang)
            split_for_nonsense(lang, args.pseudodir, args.savedir, type="ort")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Prepare feature tests
    parser.add_argument('--nonlabelratio', type=float, default=0.3)
    parser.add_argument('--savedir', type=str, default='./probing_datasets_2')
    parser.add_argument('--feat', type=int, default=0)
    parser.add_argument('--common', type=int, default=1)
    parser.add_argument('--pseudo', type=int, default=1)
    parser.add_argument('--pseudodir', type=str, default='./generated_wuggy_files')

    args = parser.parse_args()
    main(args)
