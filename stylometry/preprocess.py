#!/usr/bin/env

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm, preprocessing
import glob
from string import punctuation
from scipy import stats
from collections import Counter
from word_counter import mfw_counter

# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
# Make raw counts of the words in our texts

def vectorize(folder_location, sample_length, feat_amount, invalid_words):

    authors = []
    titles = []
    texts = []

    for filename in glob.glob(folder_location + "/*"):
    author = filename.split("/")[-1].split(".")[0].split("_")[0]
    title = filename.split("/")[-1].split(".")[0].split("_")[1]

    bulk = []

    fob = open(filename)
    text = fob.read()
    for word in text.rstrip().split():
        for char in word:
        if char in punctuation:
            word = word.replace(char, "")
        word = word.lower()
        bulk.append(word)

    bulk = [bulk[i:i+sample_length] for i in range(0, len(bulk), sample_length)]

    for index, sample in enumerate(bulk):
        if len(sample) == sample_length:
        authors.append(author)
        titles.append(title + "_{}".format(str(index + 1)))
        texts.append(" ".join(sample))

    raw_counts, features = mfw_counter(texts, feat_amount, invalid_words)

    return authors, titles, texts, raw_counts, features

# -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
# The amount of standard deviations away from the overall mean, standard Delta approach

def zscore_delta(raw_counts, authors, titles, test_dict, test_train_split):

    zscore_vectors = stats.zscore(raw_counts)
    labels = 0
    return zscore_vectors

def tfidf_vectorizer(raw_counts, authors, titles, features, test_dict, test_train_split):

    # For each feature, observe its frequency in a document
    # Normalize by document frequency

    tfidf_vectors = []

    for index, function_word in enumerate(features):
    counter = 0
    for i in raw_counts[:,index]:
        if i != 0:
        counter += 1
    tfidf_vectors.append(list(raw_counts[:,index] / counter))

    tfidf_vectors = np.array(tfidf_vectors)
    tfidf_vectors = np.transpose(tfidf_vectors)
    tfidf_vectors = tfidf_vectors / np.std(tfidf_vectors, axis=0)

    return tfidf_vectors