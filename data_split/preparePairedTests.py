# -*- coding: utf-8 -*-
import pickle
from data_util.schema import *
from data_util.reader import *

import random
import argparse
import numpy

from data_split.util import *
missing_feats = ['Possession']

######################
# Balancing datasets #
######################
# 	        Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
#  German	1349	28387	16345	19555	4919	3699	0	0	0	0
odd_sampling = {}
odd_sampling['german'] = {'Mood':1., 'Number':0.3, 'Person':0.4, 'Case':0.4, 'Tense':1., 'lemma': 1., 'Voice':0., \
                          'Gender': 0., 'Possession':0., 'Polarity':0.}

# 	        Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
# Finnish   5220	19267	13113	23891	875	    9500	231	 0	0	0
odd_sampling['finnish'] = {'Mood':1., 'Number':0.6, 'Person':0.8, 'Case':0.5, 'Tense':1., 'lemma': 1., 'Voice':1., \
                          'Gender': 0., 'Possession':0., 'Polarity':0.}

#           Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
# Spanish	1668	4750	21196	0	    4124	6500	0	   10574	0	0
odd_sampling['spanish'] = {'Mood':1., 'Number':1, 'Person':0.5, 'Case':0., 'Tense':1., 'lemma': 1., 'Voice':0., \
                          'Gender': .8, 'Possession':0., 'Polarity':0.}

#               Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
#    Russian	0	   16414	40249	7082	724	    7968	4342	24686	    0	       0
odd_sampling['russian'] = {'Mood':1., 'Number':0.3, 'Person':0.2, 'Case':0.7, 'Tense':1., 'lemma': 0.8, 'Voice':1., \
                          'Gender': 0.2, 'Possession':0., 'Polarity':0.}

#               Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
#       Turkish	   0	4342	 2877	18613	7407	17392	   0	   0	    4635	389
odd_sampling['turkish'] = {'Mood':0., 'Number':1., 'Person':1., 'Case':0.6, 'Tense':0.9, 'lemma': 0.6, 'Voice':0., \
                          'Gender': 0., 'Possession':1., 'Polarity':1.}

odd_sampling['armenian'] = {'Case':0.5, 'Possession':.5}

odd_sampling['serbo-croatian'] = {'Number':0.25, 'Person':.2, 'Case':.1, 'Gender': 0.4}

odd_sampling['italian'] = {'Person':0.3}

odd_sampling['polish'] = {'Number':0.5, 'Case':.1}

odd_sampling['french'] = {'Person':0.3}

odd_sampling['czech'] = {'Case':.3}

odd_sampling['hungarian'] = {'Case':.05}


# 	        Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
#  German	11683	62188	8629	19789	24505	1464	0	0	0	0
same_sampling = {}
same_sampling['german'] = {'Mood':1., 'Number':0.15, 'Person':1., 'Case':0.6, 'Tense':0.5, 'lemma': 1., 'Voice':0., \
                          'Gender': 0., 'Possession':0., 'Polarity':0.}

# 	        Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
# Finnish   0	   74724	   0	5427	9814	  679	 9508	   0	       0	12223
same_sampling['finnish'] = {'Mood':1., 'Number':0.2, 'Person':0., 'Case':1, 'Tense':1., 'lemma': 1., 'Voice':1., \
                          'Gender': 0., 'Possession':0., 'Polarity':1.}

#           Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
# Spanish	11862	30630	14228	   0	14402	1896	0	0	0	0
same_sampling['spanish'] = {'Mood':1., 'Number':0.5, 'Person':1., 'Case':0., 'Tense':1., 'lemma': 1., 'Voice':0., \
                          'Gender': .8, 'Possession':0., 'Polarity':0.}

#               Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
#    Russian	0	   68485	18014	12083	41074	937  	42714	   0	   0	0
same_sampling['russian'] = {'Mood':0., 'Number':0.2, 'Person':0.9, 'Case':1., 'Tense':0.3, 'lemma': 1., 'Voice':0.3, \
                          'Gender': 0.2, 'Possession':0., 'Polarity':0.}

#               Mood	Number	Person	Case	Tense	Lemma	Voice	Gender	Possession	Polarity
#       Turkish	   0	63698	7398	11584	   0	15115	   0	   0	15087	14895
same_sampling['turkish'] = {'Mood':0., 'Number':0.2, 'Person':1., 'Case':1., 'Tense':0., 'lemma': 1., 'Voice':0., \
                          'Gender': 0., 'Possession':1., 'Polarity':1.}

same_sampling['bulgarian'] = {'Mood':0.7, 'Number':0.25, 'Person':1., 'Tense':0.6, }

same_sampling['armenian'] = {'Number':0.2, 'Possession':.4}

same_sampling['portuguese'] = {'Number':0.5}

same_sampling['serbo-croatian'] = {'Number':0.5}

same_sampling['italian'] = {'Number':0.5}

same_sampling['polish'] = {'Number':0.3, 'Tense':.5}

same_sampling['french'] = {'Number':0.6}

same_sampling['dutch'] = {'Number':0.6}

same_sampling['czech'] = {'Number':0.2}

same_sampling['modern-greek'] = {'Number':0.2}

same_sampling['catalan'] = {'Number':0.5}

same_sampling['estonian'] = {'Number':0.15}

same_sampling['romanian'] = {'Number':0.3}

same_sampling['arabic'] = {'Number':0.6}

same_sampling['hungarian'] = {'Number':0.02}

same_sampling['swedish'] = {'Voice':0.5}

same_sampling['macedonian'] = {'Number':0.5}

same_sampling['quechua'] = {'Number':0.2}

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def reverse_dict_list(orig_dict):
    rev_dict = dict()
    for test in orig_dict:
        for lang in  orig_dict[test]:
            if lang not in rev_dict:
                rev_dict[lang] = [test]
            else:
                rev_dict[lang].append(test)
    return rev_dict

def check_vals_except(i_feats, j_feats, exc_feat):
    if exc_feat in missing_feats:
        if abs(len(i_feats)-len(j_feats))>1:
            return False
    else:
        if len(i_feats)!=len(j_feats):
            return False

    for feat in i_feats:
        if feat==exc_feat:
            continue
        if feat not in j_feats:
            return False
        if i_feats[feat]!=j_feats[feat]:
            return False

    return True

# Mood is for Turkish and Finnish
def check_vals_incl(i_feats, j_feats, inc_feat):
    for feat in i_feats:
        # for Turkish
        # if feat in [inc_feat, 'Part of Speech', 'Mood', 'Interrogativity']:
        # for Finnish (Person fails)
        # if feat in [inc_feat, 'Part of Speech', 'Mood']:
        # for Spanish (Gender, Polarity fails), spanish gender is always fem,pst etc...
        # for Russian Gender fails, because it is always annotated with singular
        if feat in inc_feat:
            continue
        if (feat in j_feats) and (i_feats[feat]==j_feats[feat]):
            return False
    return True

def shuffle_and_write(instances, threshold, savedir, lang, feat = "Feature"):
    shuffled_instances = random.sample(instances, threshold)
    train_inst = shuffled_instances[:int(threshold*0.7)]
    dev_inst = shuffled_instances[int(threshold*0.7):int(threshold*0.9)]
    test_inst = shuffled_instances[int(threshold*0.9):]

    train_path = os.path.join(savedir, feat, lang, "train.txt")
    ensure_dir(train_path)
    dev_path = os.path.join(savedir, feat, lang, "dev.txt")
    ensure_dir(dev_path)
    test_path = os.path.join(savedir, feat, lang, "test.txt")
    ensure_dir(test_path)

    # Write file
    with open(train_path, 'w') as fout:
        for w1, w2, label in train_inst:
            fout.write("\t".join([w1, w2, label])+"\n")
    fout.close()

    with open(dev_path, 'w') as fout:
        for w1, w2, label in dev_inst:
            fout.write("\t".join([w1, w2, label])+"\n")
    fout.close()

    with open(test_path, 'w') as fout:
        for w1, w2, label in test_inst:
            fout.write("\t".join([w1, w2, label])+"\n")
    fout.close()

def eliminate_rare(instances, vocab, keep_rat = 2):
    """
    keep it even if rare with p=0,2
    :param instances:
    :param vocab:
    :param keep_rat:
    :return:
    """
    elec_instance = []
    for w1, w2, label in instances:
        if (w1 in vocab) and (w2 in vocab):
            elec_instance.append((w1,w2,label))
        else:
            # roll the dice, leave it to luck
            num = numpy.random.choice(numpy.arange(1, 11))
            if(num<=keep_rat):
                elec_instance.append((w1, w2, label))
    return elec_instance

########################
# ODD FEATS
########################

def find_surface_pairs_w_odd_morph_feat(data, feat, lemma_sample=10):
    """
    They should have the same lemma, but only one different feature

    Examples:
    gülüşünün	gülüşlerinin	number
    günlerimize	günlerimizi	case
    yapışacak yapışır tense
    yapıştınız	yapıştık	person
    götürmeyecek	götürmeyecek	polarity
    gününüzün   günün possesion

    :param data: unimorph data
    :param feat: morph feat
    :return: prepared instances
    """
    schema = UnimorphSchema()
    group_by_lemma = {}

    # instances[0] = (w1, w2, label)
    instances = []
    print("Preparing "+feat)

    # group instances by lemma
    for x in data:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue
        x_feats = schema.decode_msd(x["msd"])[0]
        if feat in x_feats:
            # put the surface forms with the same lemma into the same cluster
            # if they have a value of the feature we are interested in, e.g., Case
            if x["lemma"] not in group_by_lemma:
                group_by_lemma[x["lemma"]] = [x]
            else:
                group_by_lemma[x["lemma"]].append(x)

    for lemma in group_by_lemma:
        x_inst_w_lemma = group_by_lemma[lemma]
        if(len(x_inst_w_lemma))==1:
            continue
        if(len(x_inst_w_lemma)>lemma_sample):
            # randomly choose 10 of them
            x_inst_w_lemma = random.sample(x_inst_w_lemma, lemma_sample)
        for i in range(len(x_inst_w_lemma)):
            for j in range(i+1,len(x_inst_w_lemma)):
                # compare the i_th and the j_th instances
                i_feats = schema.decode_msd(x_inst_w_lemma[i]["msd"])[0]
                j_feats = schema.decode_msd(x_inst_w_lemma[j]["msd"])[0]
                # bug: if person feature, add the number
                if feat == 'Person':
                    i_feats[feat] = i_feats[feat] + " " + i_feats['Number']
                    j_feats[feat] = j_feats[feat] + " " + j_feats['Number']
                    # hack the number feature
                    i_feats['Number'] = 'dumm'
                    j_feats['Number'] = 'dumm'
                # check if the feature values are different and everything else is the same OR
                # it is a missing feature and one of the items don't have that value, but all others are the same
                if ((i_feats[feat]!=j_feats[feat]) and check_vals_except(i_feats, j_feats, feat)) or \
                   ((feat in missing_feats) and ((feat not in i_feats) or (feat not in j_feats)) and \
                    check_vals_except(i_feats, j_feats, feat)):
                    # check if everything else is the same
                    tr_triple = (x_inst_w_lemma[i]['form'], x_inst_w_lemma[j]['form'], feat)
                    # still there are some errors
                    if (x_inst_w_lemma[i]['form'] == x_inst_w_lemma[j]['form']):
                        continue
                    # Check if a duplicate exists
                    if tr_triple not in instances:
                        instances.append(tr_triple)
    return instances

def find_surface_pairs_w_odd_lemma(data, sample_size = 100):
    """
    Examples: Only the lemma will be different. All other tags should be the same
    grubumuzda	göbeğimizde	lemma
    gübrenin	gülüşünün	lemma
    :param data: unimorph data
    :param sample_size: number of items with exactly same morph features
    :return: instances with odd lemma e.g., instances[0] = (w1, w2, "lemma")
    """
    group_by_feats = {}

    # instances[0] = (w1, w2, label)
    instances = []
    print("Preparing lemma")
    for x in data:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue
        if x["msd"] not in group_by_feats:
            group_by_feats[x["msd"]] = [x]
        else:
            group_by_feats[x["msd"]].append(x)

    print("Number of feature combinations "+str(len(group_by_feats)))
    for feat_comb in group_by_feats:
        x_inst_w_feats = group_by_feats[feat_comb]
        half_len = int(len(x_inst_w_feats)/2)
        frst_half = x_inst_w_feats[:half_len]
        scnd_half = x_inst_w_feats[half_len:]

        if(len(frst_half)>sample_size):
            # sample some amount
            frst_half = random.sample(frst_half, sample_size)
            scnd_half = random.sample(scnd_half, sample_size)

        # now just prepare the data
        for i in range(len(frst_half)):
            tr_triple = (frst_half[i]['form'], scnd_half[i]['form'], "lemma")
            # Check if a duplicate exists
            if tr_triple not in instances:
                instances.append(tr_triple)
    return instances

def split_for_odd_feature(lang, test_lst, savedir, vocab, threshold=10000):
    '''
    Split the dataset for language
    :param lang: the language
    :param test_lst: the list of features that can be probed
    :param vocab: frequency list
    :param savedir: where to save the test files
    :return: false if there was a problem splitting
    '''
    # Get the data for language
    data = load_ds("unimorph", lang)
    instances = []

    for test_name in test_lst:
        # Find the word pairs that are different i.t.o. test_name
        if (lang in ['finnish','russian']) and (test_name in ["Case"]):
            lemma_sample = 2
        elif (lang in ['finnish','russian']):
            lemma_sample = 5
        else:
            lemma_sample = 10
        odd_words = find_surface_pairs_w_odd_morph_feat(data[lang], test_name, lemma_sample=lemma_sample)

        # Sample, according to the guidelines
        if (lang in odd_sampling) and (test_name in odd_sampling[lang]):
            odd_words = random.sample(odd_words, int(len(odd_words)*odd_sampling[lang][test_name]))
        else:
            odd_words = random.sample(odd_words, len(odd_words))
        print(len(odd_words))
        instances += odd_words

    # Find the word pairs that are different w.r.t lemma
    odd_words = find_surface_pairs_w_odd_lemma(data[lang])
    if (lang in odd_sampling) and ('lemma' in odd_sampling[lang]):
        odd_words = random.sample(odd_words, int(len(odd_words) * odd_sampling[lang]['lemma']))
    else:
        odd_words = random.sample(odd_words, len(odd_words))
    print(len(odd_words))
    instances += odd_words

    # Eliminate the rare ones
    instances = eliminate_rare(instances, vocab, keep_rat=2)

    # make a more balanced dataset ?
    if len(instances) < threshold:
        print(str(len(instances))+" are left")
        return False

    # shuffle and write

    shuffle_and_write(instances, threshold, savedir, lang, "OddFeat")
    return True

########################
# SAME FEATS
########################

def find_surface_pairs_w_same_morph_feat(data, feat, lang, feat_sample=500):
    """
    They should have the same value for feat, but everything else should be different

    Examples:
    gölgemizde gölgelerin lemma
    gölgelerimizi gömleklerine number

    :param data: unimorph data
    :param feat: morph feat
    :param lang: some language specific features
    :return: prepared instances
    """
    schema = UnimorphSchema()

    # pair_instances[0] = (w1, w2, label)
    pair_instances = []

    instances = []
    print("Preparing "+feat)

    inc_feat_for_lang = [feat, 'Part of Speech']
    # for Turkish
    # if feat in [inc_feat, 'Part of Speech', 'Mood', 'Interrogativity']:
    # for Finnish (Mood, Person fails)

    if lang in ['turkish','finnish']:
        inc_feat_for_lang.append('Mood')
    if lang=='turkish':
        inc_feat_for_lang.append('Interrogativity')

    # group instances by feature
    for x in data:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue
        x_feats = schema.decode_msd(x["msd"])[0]
        if feat in x_feats:
            instances.append(x)

    # make a sample from instances with feature
    if(len(instances)<feat_sample):
        print("Not enough instances")
        return False

    shuffled_instances = random.sample(instances, len(instances))
    sample_instances = shuffled_instances[:feat_sample]
    comp_instances = shuffled_instances[feat_sample:(2*feat_sample)]

    for sample_inst in sample_instances:
        for comp_inst in comp_instances:
            # lemma is different for sure
            if sample_inst["lemma"]==comp_inst["lemma"]:
                continue
            # compare the i_th and the j_th instances
            i_feats = schema.decode_msd(sample_inst["msd"])[0]
            j_feats = schema.decode_msd(comp_inst["msd"])[0]
            if feat == 'Person':
                i_feats[feat] = i_feats[feat] + " " + i_feats['Number']
                j_feats[feat] = j_feats[feat] + " " + j_feats['Number']
                # hack the number feature
                i_feats['Number'] = 'dumm1'
                j_feats['Number'] = 'dumm2'
            # check if the feature values are the same and everything else is different
            if ((i_feats[feat]==j_feats[feat]) and check_vals_incl(i_feats, j_feats, inc_feat_for_lang)):
                tr_triple = (sample_inst['form'], comp_inst['form'], feat)
                # still there are some errors
                if (sample_inst['form'] == comp_inst['form']):
                    continue
                # Check if a duplicate exists
                if tr_triple not in pair_instances:
                    pair_instances.append(tr_triple)
    return pair_instances

def find_surface_pairs_w_same_lemma(data, feat_sample=100):
    """
    They should have the same lemma, but all other features should be different

    :param data: unimorph data
    :param feat: morph feat
    :return: prepared instances
    """
    group_by_feats = {}

    # instances[0] = (w1, w2, label)
    instances = []
    print("Preparing lemma")

    # group instances by lemma
    for x in data:
        # exclude lemmas with space
        if ' ' in x["form"]:
            continue
        # put the surface forms with the same lemma into the same cluster
        if x["msd"] not in group_by_feats:
            group_by_feats[x["msd"]] = [x]
        else:
            group_by_feats[x["msd"]].append(x)
    print("Number of different groups: "+str(len(group_by_feats)))
    for i in range(int(len(group_by_feats))):
        randkey1 = random.choice(list(group_by_feats))
        randkey2 = random.choice(list(group_by_feats))
        if(randkey1 == randkey2):
            continue
        # sample from first group
        sample_size = min(len(group_by_feats[randkey1]), len(group_by_feats[randkey2]), feat_sample)
        first_instances = random.sample(group_by_feats[randkey1], sample_size)
        second_instances = random.sample(group_by_feats[randkey2], sample_size)
        for inst1 in first_instances:
            for inst2 in second_instances:
                if(inst1["lemma"]==inst2["lemma"]):
                    tr_triple = (inst1['form'], inst2['form'], "lemma")
                    # still there are some errors
                    if (inst1['form'] == inst2['form']):
                        continue
                    # Check if a duplicate exists
                    if tr_triple not in instances:
                        instances.append(tr_triple)
    return instances

def split_for_same_feature(lang, test_lst, savedir, vocab, threshold=10000):
    '''
    Split the dataset for language
    :param lang: the language
    :param test_lst: the list of features that can be probed
    :param vocab: frequency list
    :param savedir: where to save the test files
    :return: false if there was a problem splitting
    '''
    # Get the data for language
    data = load_ds("unimorph", lang)
    instances = []
    feat_sample = 400

    for test_name in test_lst:
        # Find the word pairs that are same i.t.o. test_name
        same_words = find_surface_pairs_w_same_morph_feat(data[lang], test_name, lang, feat_sample=feat_sample)
        # Sample, according to the guidelines
        if (lang in same_sampling) and (test_name in same_sampling[lang]):
            same_words = random.sample(same_words, int(len(same_words)*same_sampling[lang][test_name]))
        else:
            same_words = random.sample(same_words, len(same_words))
        print(len(same_words))
        instances += same_words

    # Find the word pairs that are different w.r.t lemma
    same_words = find_surface_pairs_w_same_lemma(data[lang], feat_sample=400)
    if (lang in same_sampling) and ('lemma' in same_sampling[lang]):
        same_words = random.sample(same_words, int(len(same_words) * same_sampling[lang]['lemma']))
    else:
        same_words = random.sample(same_words, len(same_words))
    print(len(same_words))
    instances += same_words

    # Eliminate the rare ones
    instances = eliminate_rare(instances, vocab, keep_rat=2)

    # make a more balanced dataset ?
    if len(instances) < threshold:
        print(str(len(instances))+" are left")
        return False

    # shuffle and write
    shuffle_and_write(instances, threshold, savedir, lang, "SameFeat")
    return True


def main(args):

    langs = {'portuguese': 'pt',
                    'french': 'fr',
                    'serbo-croatian': 'sh',
                    'polish': 'pl',
                    'czech':'cs',
                    'modern-greek':'el',
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
                    'romanian':'ro',
                    'ukranian': 'uk',
                    'german': 'de',
                    'finnish': 'fi',
                    'russian': 'ru',
                    'turkish': 'tr',
                    'spanish': 'es'
                   }

    # Language specific vocabulary sizes
    # wiki vocabulary sizes: de: 2275234, es: 985668, fi: 730484, tr: 416052, ru: 1888424

    langs_vocab = {'german': 750000, 'finnish':500000, 'russian': 750000, 'turkish':500000, 'spanish': 500000,
                         'portuguese':500000, 'french':750000, 'serbo-croatian':500000, 'polish':750000, 'czech':500000,
                         'modern-greek':500000, 'catalan':500000, 'bulgarian':500000, 'danish': 500000, 'estonian': 500000,
                         'quechua': 500000, 'swedish': 750000, 'armenian': 500000, 'macedonian': 500000, 'arabic': 500000,
                          'dutch': 600000, 'hungarian': 600000, 'italian': 600000, 'romanian':500000, 'ukranian': 750000}

    with open('test_vs_lang_feat_over_10K.pkl', 'rb') as handle:
        test_vs_lang = pickle.load(handle)

    lang_vs_test = reverse_dict_list(test_vs_lang)

    for lang in lang_vs_test:
        if lang in langs:
            embfile = os.path.join('..', "embeddings", "wiki." + langs[lang] + ".vec")
            print("Reading vocabulary for lang "+lang)
            vocab = load_dict(embfile, maxvoc=langs_vocab[lang])
            print("Preparing odd feature test for " + lang)
            split_for_odd_feature(lang, lang_vs_test[lang], args.savedir, vocab)
            print("Preparing same feature test for " + lang)
            split_for_same_feature(lang, lang_vs_test[lang], args.savedir, vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Prepare feature tests
    parser.add_argument('--keepratio', type=float, default=0.2)
    parser.add_argument('--savedir', type=str, default='./probing_tests')
    args = parser.parse_args()
    main(args)