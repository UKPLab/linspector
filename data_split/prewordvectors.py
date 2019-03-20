#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Author: Gözde Gül Şahin
Load pretrained embeddings
DEPRECATED!
INSTEAD USE FASTTEXT WITH GENSIM:


from gensim.models import FastText as fText

'''
 There are two files in directory "/home/jack/dev1.8t/models/vecs/":

      zhwiki15-100_200dim.vec
      zhwiki15-100_200dim.bin
'''
filename = '/home/sahin/Workspace/Projects/invertible_NNs/datasets/nlp/wiki.tr/wiki.tr'
fastText_wv = fText.load_fasttext_format(filename)
fastText_wv.wv.most_similar("katılamayanlardan")



# Find the top-N most similar words by vector.
fastText_wv.similar_by_vector()


fasttext lib can also be used
import fastText
binfile = '/home/sahin/Workspace/Projects/invertible_NNs/datasets/nlp/wiki.tr/wiki.tr.bin'
vecfile = '/home/sahin/Workspace/Projects/invertible_NNs/datasets/nlp/wiki.tr/wiki.tr.vec'

model = fastText.load_model(binfile)
model.get_word_vector('katilamayanlardanmisin')
"""

def load_dict(embfile, embsize = 300, maxvoc=None):
    """
    Load the vocabulary from sorted embedding files
    :param embfile: word embedding file
    :param embsize: word vector size
    :param maxvoc: maximum vocabulary size
    :return: dictionary (numpy matrix of shape (num_embeddings, embedding_dim))
    """
    word_to_ix = {}

    # fill unk word with random numbers
    f = open(embfile, 'r')
    ix = 0
    for line in f:
        if (maxvoc!=None) and (ix >= maxvoc):
            break
        splitLine = line.split()
        if(len(splitLine)>embsize+1):
            phrase_lst = splitLine[:-embsize]
            word = ' '.join(phrase_lst)
            word_to_ix[word] = ix
            ix += 1
        elif(len(splitLine)>2):
            word = splitLine[0]
            word_to_ix[word]=ix
            ix += 1

    #print("%d words loaded!" % len(word_to_ix))
    return word_to_ix
