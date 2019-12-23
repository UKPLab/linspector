import sys

sys.path.append('../')

import argparse
import os

import torch
from elmoformanylangs import Embedder

from pytorch_transformers import BertModel, BertTokenizer
from tqdm import tqdm

import numpy as np

from data_util import io_helper

"""
Creates static vectors for vocabulary
"""


def get_elmo(model, sentence_list, indices_list):
    """
    Get elmo embeddings
    :param model: elmo model
    :param sentence_list: List of sentences for which embeddings will be created
    :param indices_list: List of list of indices. For every sentence: A list of indices for which words embeddings will be created
    :return: List of list of word vectors. First list represents the sentences. Second list contains the word vectors in the same order as the indices were given in the indices_list.
    """
    vector_list = []

    # Give ELMo a list of sentences
    embedding = model.sents2elmo(sentence_list)
    for i in range(0, len(sentence_list)):
        # Extract the words matching the indices for this sentence
        indices = indices_list[i]
        for ind in indices:
            if ind == "":
                continue
            ind = int(ind)

            e = embedding[i][ind]
            vector_list += [e.tolist()]
    return vector_list


def load_bert():
    """
    Loads BERT model
    :return: bert model
    """
    pretrained_weights = 'bert-base-multilingual-cased'
    model = BertModel.from_pretrained(pretrained_weights)
    # Move BERT to cuda if available
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def get_bert(model, tokenizer, sentence_list, index_list):
    """
    Get BERT embeddings
    :param model: BERT model
    :param tokenizer: BERT tokenizer
    :param sentence_list: List of sentences for which embeddings will be created
    :param indices_list: List of list of indices. For every sentence: A list of indices for which words embeddings will be created
    :return: List of list of word vectors. First list represents the sentences. Second list contains the word vectors in the same order as the indices were given in the indices_list.
    """
    vector_list = []

    for i in range(0, len(sentence_list)):
        sentence = sentence_list[i]
        indices = index_list[i]
        sentence_str = " ".join(sentence)
        if torch.cuda.is_available():
            input_ids_sentence = torch.tensor([tokenizer.encode(sentence_str, add_special_tokens=True)], device="cuda")
        else:
            input_ids_sentence = torch.tensor([tokenizer.encode(sentence_str, add_special_tokens=True)])
        last_hidden_states = model(input_ids_sentence)[0]  # Models outputs are now tuples

        # Find relevant vectors
        token_count_per_word = []
        for j in range(0, len(sentence)):
            # Save number of tokens per word
            word = sentence[j]
            tokens = tokenizer.encode(word, add_special_tokens=False)
            token_count_per_word.append(len(tokens))

        for j in range(0, len(indices)):
            if indices[j] == "":
                continue
            # Find tokens for this word
            index = int(indices[j])
            offset = 1
            for k in range(0, index):
                offset += token_count_per_word[k]
            last_token = offset + token_count_per_word[index]
            vector_list += [np.mean(last_hidden_states[0][offset:last_token].tolist(),
                                    axis=0)]  # BERT uses word piece tokenization. Calculate the average over all tokens from this word

    return vector_list


def load_elmo(lang):
    """
    Loads elmo model
    :param lang: language of the model
    :return: elmo model
    """
    emb = Embedder("../embeddings/elmo/" + lang)
    return emb


def gen_embeds(vocab_dir, output_dir, lang):
    """
    Generates all supported embeddings for given language
    :param vocab_dir: Directory of the vocabulary files
    :param output_dir: Directory of the embedding files
    :param lang: language
    """
    print("Generate " + lang + " embeddings")
    # Read the input file to a list
    sentence_file = os.path.join(vocab_dir, lang + ".txt")
    with open(sentence_file) as f:
        lines = f.read().splitlines()
    f.close()

    # Delete embeddings
    clear_embeds(output_dir, "elmo", lang)
    clear_embeds(output_dir, "bert", lang)

    sentence_list = []
    indices_list = []
    for line in lines:
        parts = line.split("\t")
        tokens = parts[0].split(" ")
        indices = parts[1].split(" ")
        sentence_list.append(tokens)
        indices_list.append(indices)

    # Batch
    batch = 10
    s = int(len(sentence_list) / batch) + 1
    # Load BERT
    bert = load_bert()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    for i in tqdm(range(0, s)):
        # Check if this is the last batch and correct the last index
        start = i * batch
        end = (i + 1) * batch
        if start >= len(sentence_list):
            break
        if end > len(sentence_list):
            end = len(sentence_list)

        # Sentences and indices for this batch
        sl = sentence_list[start:end]
        il = indices_list[start:end]

        # Extract and save embeddings
        store_embeds(output_dir, sl, il, get_elmo(load_elmo(lang), sl, il), lang, "elmo")
        store_embeds(output_dir, sl, il, get_bert(bert, bert_tokenizer, sl, il), lang, "bert")


def clear_embeds(output_dir, embedding, lang):
    """
    Deletes all embedding files for the given language and embedding
    :param output_dir: directory of the embeddings
    :param embedding: embedding
    :param lang: language
    """
    outfile = os.path.join(output_dir, embedding, lang + ".vec")
    if os._exists(outfile):
        os.remove(outfile)


def store_embeds(output_dir, sentence_list, indices_list, vector_list, lang, embedding):
    """
    Store all embeddings in a file
    :param output_dir: Directory of the embedding files
    :param sentence_list: list of list of tokens
    :param indices_list: list of list of word indices.
    :param vector_list: list of word vectors for the tokens
    :param lang: language
    :param embedding: embedding type
    """
    # Create the output file
    outfile = os.path.join(output_dir, embedding, lang + ".vec")
    io_helper.ensure_dir_for_file(outfile)

    with open(outfile, 'a') as fout:
        j = 0
        for i in range(0, len(sentence_list)):
            # Write a separate line for every sentence index-pair
            sentence = sentence_list[i]
            sentence_str = " ".join(sentence)
            for index in indices_list[i]:
                if index == "":
                    continue
                # Read vector and convert it to a string
                vector = vector_list[j]
                j += 1
                vector_string = ""
                z = 0
                for number in vector:
                    if z != 0:
                        vector_string += " "
                    z += 1
                    vector_string += str(number)
                # Write vector for sentence-index pair to the file
                fout.write(sentence_str + "\t" + index + "\t" + vector_string + "\n")
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # argument options
    parser.add_argument('--vocab', type=str, required=False, default="../intrinsic/sentences/",
                        help='directory of the vocabulary')
    parser.add_argument('--output', type=str, required=False, default="../intrinsic/static_context_embeddings/",
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
        print("Language " + lang)
        gen_embeds(args.vocab, args.output, lang)
