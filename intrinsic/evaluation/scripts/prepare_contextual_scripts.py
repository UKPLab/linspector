import sys

sys.path.append('../../../')

import argparse
import os
import pickle

from data_util.io_helper import ensure_dir_for_file, ensure_dir

embedding_dir = "../static_embeddings/"
embedding_dir_c = "../static_context_embeddings/"


def contains_uncased(_list, string):
    for item in _list:
        if item.lower() == string.lower():
            return True
    return False


def reverse_dict_list(orig_dict):
    rev_dict = dict()
    for test in orig_dict:
        for lang in orig_dict[test]:
            if lang not in rev_dict:
                rev_dict[lang] = [test]
            else:
                rev_dict[lang].append(test)
    return rev_dict


def create_config_file(feat, lang, contextualized, embedding, emb_dim, gpu, test_vs_lang_vs_values, models_dir,
                       template_dir, config_dir):
    """
    Creates config file for model
    :param feat: feature
    :param lang: language
    :param contextualized: True if model should train with context
    :param embedding: embedding type (bpe or elmo)
    :param gpu: index of cuda device. -1 if cpu should be used
    :param test_vs_lang_vs_values: Dictionary feat vs language vs count of feature values for this language and feature
    :return: [path to config file, folder for model]
    """
    c = ""
    if contextualized:
        c = "_c"
    mid = lang + "_" + feat.replace(" ", "_") + "_" + embedding + "_" + c
    mid = mid.lower()
    output_file = os.path.join("..", config_dir, mid + ".json")
    output_file = output_file.lower()
    ensure_dir_for_file(output_file)
    fout = open(output_file, 'w')

    if contextualized:
        template = os.path.join(template_dir, "contextual.json")
    else:
        template = os.path.join(template_dir, "non_contextual.json")

    feat_values = test_vs_lang_vs_values[feat][lang]
    values_count = len(feat_values)

    # Write new config file based on template
    with open(template) as f:
        for line in f:
            newline = line.replace("LANG_LONG", lang).replace("FEAT", feat).replace(
                "GPU", str(gpu)).replace("XXX", str(values_count)).replace("EMB_DIM", str(emb_dim))
            if not contextualized:
                embedding_file = embedding_dir + embedding + "/" + lang + ".vec"
            else:
                embedding_file = embedding_dir_c + embedding + "/" + lang + ".vec"
            newline = newline.replace("pretrained_embedding_file", embedding_file)
            fout.write(newline)
        fout.close()
    ensure_dir(os.path.join("..", models_dir))
    return [os.path.join(config_dir, mid + ".json"), os.path.join(models_dir, mid)]


if __name__ == "__main__":
    with open('../../data_contextual/features.pkl', 'rb') as handle:
        test_vs_lang_vs_values = pickle.load(handle)

    parser = argparse.ArgumentParser()

    # argument options
    parser.add_argument('--gpu', type=int, required=False, default=0,
                        help='gpu device. -1 to use cpu')
    parser.add_argument('--lang', type=str, required=False, default=None)
    parser.add_argument('--static_embedding', type=str, required=False, default=None)
    parser.add_argument('--contextual_embedding', type=str, required=False, default=None)
    parser.add_argument('--task', type=str, required=False, default=None)
    parser.add_argument('--models', type=str, required=False, default='models/')
    parser.add_argument('--script', type=str, required=False, default='train.sh')
    parser.add_argument('--templates', type=str, required=False, default='config_templates')
    parser.add_argument('--configs', type=str, required=False, default='configs')

    args = parser.parse_args()

    # Relative path to model config files
    configs = []

    # Supported languages and embeddings
    langs = ["german", "finnish",
             "spanish", "turkish", "russian"]
    # Embeddings vs dim
    embeddings_uncont = {"bpe": 300, "fasttext": 300, "muse": 300, "w2v": 300}
    embeddings = {"bert": 768, "elmo": 1024}

    # Parse arguments
    tasks = None
    if args.task is not None:
        tasks = args.task.split(',')

    if args.lang is not None:
        langs = args.lang.split(",")

    a_con_emb = None
    if args.contextual_embedding is not None:
        a_con_emb = args.contextual_embedding.split(",")

    a_stat_emb = None
    if args.static_embedding is not None:
        a_stat_emb = args.static_embedding.split(",")

    # Create config file for every supported language, feature and embedding
    for feat in test_vs_lang_vs_values:
        if tasks is not None and feat not in tasks:
            continue
        for lang in test_vs_lang_vs_values[feat]:
            if langs is not None and lang not in langs:
                continue
            for embedding in embeddings:
                if a_stat_emb is not None and a_con_emb is None:
                    break
                # Train contextual and static model if available
                for c in [True, False]:
                    if c and a_con_emb is not None and embedding not in a_con_emb:
                        continue
                    if not c and a_stat_emb is not None and embedding not in a_stat_emb:
                        continue
                    file = create_config_file(feat, lang, c, embedding, embeddings[embedding], args.gpu,
                                              test_vs_lang_vs_values,
                                              args.models, args.templates, args.configs)
                    configs.append(file)
            for embedding in embeddings_uncont:
                if a_con_emb is not None and a_stat_emb is None:
                    break
                if a_stat_emb is not None and embedding not in a_stat_emb:
                    continue
                file = create_config_file(feat, lang, False, embedding, embeddings_uncont[embedding], args.gpu,
                                          test_vs_lang_vs_values,
                                          args.models, args.templates, args.configs)
                configs.append(file)

    output_file = os.path.join("..", args.script)
    fout = open(output_file, 'w')

    # Create bash script which trains all created models
    for file in configs:
        fout.write("allennlp train " + file[0] + " -s " + file[1] + " --include-package classifiers\n")
    fout.close()
