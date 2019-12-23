import sys

sys.path.append('../')

import argparse
import os

import numpy as np
import torch
from bpemb import BPEmb
#####################################################################
# Requirements:
#               bpemb  - pip install
#               allennlp - pip install
#               https://github.com/HIT-SCIR/ELMoForManyLangs
#               pytorch-transformers - pip install pytorch-transformers
#               numpy
#####################################################################
from elmoformanylangs import Embedder
from gensim.models import FastText, KeyedVectors
from pytorch_transformers import *
from tqdm import tqdm

from data_util import io_helper
from data_util.lang_converter import convert_long_to_short

"""
Creates static vectors for vocabulary
"""


def loadBPE(lang, vector_size):
    """
    It automatically downloads the embedding file and loads it as a gensim keyed vector
    :param lang: langauge is enough, no need for embedding file
    :param vector_size: Size of the word vector
    :return:
    """

    model = BPEmb(lang=convert_long_to_short(lang), dim=vector_size)
    return model


def get_bpe(model, words):
    """
    Get BPE embeddings. It only works word by word!! Will be slow
    :param model: bpe model
    :param words: words
    :return:
    """

    vector_list = []

    for i in range(0, len(words)):
        word = words[i]
        embed = model.embed(word)
        vector = np.mean(embed, axis=0) # BPE creates multiple vectors for a word. Use the mean
        vector_list += [vector.tolist()]

    return vector_list


def get_elmo(model, words, batch_size=16):
    """
    Get elmo embeddings
    :param words: list
    :param model: elmo model
    :param batch_size: batch size (should be 16)
    :return:
    """
    vector_list = []
    # Create batches
    batch = []
    for i in range(0, len(words)):
        word = words[i]
        batch.append([word])
        if len(batch) == batch_size or i == len(words) - 1:
            # Send sentences to elmo and save the embeddings in a list
            embed = model.sents2elmo(batch)
            for i in range(0, len(embed)):
                vector_list += [embed[i][0].tolist()]
            batch = []

    return vector_list


def get_bert(words):
    """
    Get elmo embeddings
    :param words: list of words
    :return:
    """
    use_gpu = torch.cuda.is_available()
    pretrained_weights = 'bert-base-multilingual-cased'
    model = BertModel.from_pretrained(pretrained_weights)

    # Copy model to gpu
    if use_gpu:
        model.to("cuda")
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    vector_list = []

    for i in tqdm(range(0, len(words))):
        word = words[i]
        dev = "cuda"
        if not use_gpu:
            dev = "cpu"
        # Give BERT a sentence and save the embeddings in a list
        input_ids = torch.tensor([tokenizer.encode(word, add_special_tokens=True)], device=dev)
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        vector = np.mean(last_hidden_states[0][1:-1].tolist(),
                         axis=0)  # BERT uses word piece tokenization. Calculate the average over all tokens from this word
        try:
            iter(vector)
        except TypeError:
            print("Word " + word + " has illegal vector " + str(vector))
        vector_list += [
            vector]

    return vector_list


def load_elmo(lang):
    """
    Loads elmo model
    :param lang: language of the model
    :return: elmo model
    """
    return Embedder("../embeddings/elmo/" + lang)


### Loading Embedding Functions ###
def load_fasttext(lang):
    """
    Load fasttext embeddings
    :param embfile:
    :return: fasttext model
    """
    lang_short = convert_long_to_short(lang)
    embfile = "../embeddings/fasttext/cc." + lang_short + ".300.bin"
    model = FastText.load_fasttext_format(embfile)
    return model


def get_fasttext(model, words, dim):
    """
    Get fasttext embeddings
    :param model: fasttext model
    :param words: words
    :return:
    """
    # Make batches of words
    vector_list = []

    null = np.zeros(dim)

    for i in range(0, len(words)):
        word = words[i]
        try:
            embed = model[word]
        except KeyError:
            embed = null
        vector = embed
        vector_list += [vector.tolist()]

    return vector_list


def gen_embeds(vocab_dir, output_dir, lang, embeddings):
    """
    Generates all supported embeddings for given language
    :param lang: language
    """
    # Read the words from the vocab file
    with open(os.path.join(vocab_dir, lang + ".txt")) as f:
        words = f.read().splitlines()
    f.close()

    # Extract and save the embeddings for all words
    if embeddings is None or "fasttext" in embeddings:
        store_embeds(output_dir, words, get_fasttext(load_fasttext(lang), words, 300), lang, "fasttext")
    if embeddings is None or "bpe" in embeddings:
        store_embeds(output_dir, words, get_bpe(loadBPE(lang, 300), words), lang, "bpe")
    if embeddings is None or "elmo" in embeddings:
        store_embeds(output_dir, words, get_elmo(load_elmo(lang), words), lang, "elmo")
    if embeddings is None or "bert" in embeddings:
        store_embeds(output_dir, words, get_bert(words), lang, "bert")


def store_embeds(embeddings_dir, words, vector_list, lang, embedding):
    """
    Store all embeddings in a file
    :param embeddings_dir: Directoy of the embedding files
    :param words: list of tokens
    :param vector_list: list of word vectors for the tokens
    :param lang: language
    :param embedding: embedding type
    """
    # Create the output file
    outfile = os.path.join(embeddings_dir, embedding, lang + ".vec")
    io_helper.ensure_dir_for_file(outfile)

    with open(outfile, 'w') as fout:
        for i in range(0, len(words)):
            # Every word is in a separate line
            word = words[i]
            vector = vector_list[i]
            vector_string = ""
            # Convert vector to string and write it to the file
            try:
                for number in vector:
                    vector_string += str(number) + " "
                fout.write(word + " " + vector_string + "\n")
            except TypeError:
                print("Word " + word + " has illegal vector " + str(vector))
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # argument options
    parser.add_argument('--vocab', type=str, required=False, default="../intrinsic/words/",
                        help='directory of the vocabulary')
    parser.add_argument('--output', type=str, required=False, default="../intrinsic/static_embeddings/",
                        help='output directory of the embeddings')
    parser.add_argument('--embeddings', type=str, required=False, default=None)
    parser.add_argument('--langs', type=str, required=False, default=None)

    args = parser.parse_args()

    embeddings = None
    if args.embeddings is not None:
        embeddings = args.embeddings.split(",")

    langs = ["german", "finnish", "russian", "turkish", "spanish"]
    if args.langs is not None:
        langs = args.langs.split(",")
    for lang in langs:
        gen_embeds(args.vocab, args.output, lang, embeddings)
