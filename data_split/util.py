#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Loads the frequency dictionary
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
