import argparse
import os

import numpy as np
from bpemb import BPEmb
from elmoformanylangs import Embedder
from gensim.models import FastText as fText


#####################################################################
# Requirements: fasttext - install from github repo
#               gensim - latest version (3.7)
#               bpemb  - pip install
#               allennlp - pip install
#               https://github.com/HIT-SCIR/ELMoForManyLangs
#####################################################################

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


### Util Functions ###
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


### Loading Embedding Functions ###
def loadfasttext(embfile):
    """
    Load fasttext embeddings
    :param embfile:
    :return:
    """
    model = fText.load_fasttext_format(embfile)
    return model


def loadBPE(lang, vector_size):
    """
    It automatically downloads the embedding file and loads it as a gensim keyed vector
    :param lang: langauge is enough, no need for embedding file
    :return:
    """
    model = BPEmb(lang=lang, dim=vector_size)
    return model


def loadElmo(embfile):
    """

    :param embfile:
    :return:
    """
    model = Embedder(embfile)
    return model


def loadw2v(embfile):
    """
    To be filled
    :param embfile:
    :return:
    """
    return None


def loadModel(infile, lang, w2vtype, vector_size):
    """
    Load the embedding model
    :param infile: Embedding file
    :param lang: language
    :param w2vtype: type of embeddings
    :return:
    """
    if w2vtype in ['fasttext', 'muse_supervised', 'muse_unsupervised']:
        return loadfasttext(infile)
    elif w2vtype == 'bpe':
        return loadBPE(lang, vector_size)
    elif w2vtype == 'elmo':
        return loadElmo(infile)
    elif w2vtype == 'w2v':
        return loadw2v(infile)
    else:
        print("Should be one of w2v|fasttext|bpe|elmo")
        return None


def word_by_word_retr(sent, model):
    """
    Util function for fasttext retrieval
    :param sent: batch of words, just a list of words, don't have to be a sentence
    :return: numpy matric BxD
    """
    # check word by word
    first_word = sent[0]

    try:
        batch_word_embed = model[first_word]
    except:
        batch_word_embed = model[u'<UNK>']

    # initialize batch_sent_emb with the first word's embedding
    batch_sent_emb = batch_word_embed
    for i in range(1, len(sent)):
        word = sent[i]
        try:
            batch_word_embed = model[word]
        except:
            batch_word_embed = model[u'<UNK>']
        batch_sent_emb = np.vstack((batch_sent_emb, batch_word_embed))
    return batch_sent_emb


def get_fasttext(model, words, batch_size):
    """
    Get fasttext embeddings
    :param model: fasttext model
    :param words: words
    :param batch_size: batch size
    :return:
    """
    # Make batches of words
    sents = list(chunks(words, batch_size))
    try:
        batch_sent_embs = model[sents[0]]
    except:
        batch_sent_embs = word_by_word_retr(sents[0], model)

    for i in range(1, len(sents)):
        # print(i)
        sent = sents[i]
        try:
            emb_for_sent = model[sent]
        except:
            emb_for_sent = word_by_word_retr(sent, model)
        batch_sent_embs = np.concatenate((batch_sent_embs, emb_for_sent))

    return batch_sent_embs


def get_bpe(model, words):
    """
    Get BPE embeddings. It only works word by word!! Will be slow
    :param model: bpe model
    :param words: words
    :param batch_size: batch size
    :return:
    """
    # Make batches of words
    # check word by word
    first_word = words[0]
    batch_word_embed = np.mean(model.embed(first_word), axis=0)
    batch_sent_emb = batch_word_embed

    for i in range(1, len(words)):
        word = words[i]
        batch_word_embed = np.mean(model.embed(word), axis=0)
        batch_sent_emb = np.vstack((batch_sent_emb, batch_word_embed))

    return batch_sent_emb


def get_elmo(model, words, batch_size):
    """
    Get fasttext embeddings
    :param model: fasttext model
    :param words: words
    :param batch_size: batch size (should be 1)
    :return:
    """
    # Make batches of words - since it is contextual embeddings, make sentences of size 1
    sents = list(chunks(words, batch_size))
    batch_sent_embs = model.sents2elmo(sents)
    batch_sent_embs = np.vstack(batch_sent_embs)
    return batch_sent_embs


def get_w2v(model, words, batch_size):
    """
    Get fasttext embeddings
    :param model: fasttext model
    :param words: words
    :param batch_size: batch size
    :return:
    """
    # Make batches of words
    sents = list(chunks(words, batch_size))
    try:
        batch_sent_embs = model[sents[0]]
    except:
        batch_sent_embs = word_by_word_retr(sents[0])

    for i in range(1, len(sents)):
        sent = sents[i]
        try:
            emb_for_sent = model[sent]
        except:
            emb_for_sent = word_by_word_retr(sent)

        batch_sent_embs = np.concatenate((batch_sent_embs, emb_for_sent))

    return batch_sent_embs


def generateEmbeds(model, infile, outfile, w2vtype, batch_size):
    # Read the input file to a list
    with open(infile) as f:
        lines = f.read().splitlines()
    f.close()

    if w2vtype in ['fasttext', 'muse_supervised', 'muse_unsupervised']:
        nparray = get_fasttext(model, lines, batch_size)
    elif w2vtype == 'bpe':
        nparray = get_bpe(model, lines)
    elif w2vtype == 'elmo':
        nparray = get_elmo(model, lines, 1)
    elif w2vtype == 'w2v':
        nparray = get_w2v(model, lines, batch_size)
    else:
        outlines = ''

    # Create the output file
    ensure_dir(outfile)
    np.savetxt(outfile, nparray, delimiter=' ', fmt='%1.5f')  # X is an array
    return


def main(args):
    model = loadModel(args.embedding, args.lang, args.w2vtype, args.vector_size)
    outfile = os.path.join(args.savedir, args.lang, args.w2vtype, "embeds_" + str(args.part) + ".vec")
    generateEmbeds(model, args.infile, outfile, args.w2vtype, args.batch_size)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Prepare feature tests
    parser.add_argument('--w2vtype', type=str, default='fasttext', help='<w2v|fasttext|bpe|elmo|muse_supervised>')
    parser.add_argument('--lang', type=str, default="tur", help="<tr|de|ru|fi|es>")
    parser.add_argument('--embedding', type=str, default="./embeddings/wiki.tr/wiki.tr")
    parser.add_argument('--infile', type=str, default="./words/tur.txt")
    parser.add_argument('--savedir', type=str, default='./saved_embeddings')
    parser.add_argument('--part', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--vector_size', type=int, default=300)

    args = parser.parse_args()
    main(args)
