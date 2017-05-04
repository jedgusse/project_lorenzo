#!/usr/bin/env

import matplotlib.pyplot as plt
import numpy as np
import glob
from string import punctuation
from collections import Counter
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import itertools
from scipy import stats
from word_counter import mfw_counter
from preprocess import vectorize
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, chi2, f_classif, mutual_info_classif, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.base import BaseEstimator, TransformerMixin

class TermfreqVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
    pass

    def fit(X, Y):
    return X, Y

    def transform(X, Y):

    tfidf_vectors = []

    for index, function_word in enumerate(features):
        counter = 0
        for i in X[:,index]:
        if i != 0:
            counter += 1
        tfidf_vectors.append(list(X[:,index] / counter))

    tfidf_vectors = np.array(tfidf_vectors)
    tfidf_vectors = np.transpose(tfidf_vectors)
    tfidf_vectors = tfidf_vectors / np.std(tfidf_vectors, axis=0)

    return tfidf_vectors, Y

print(TermfreqVectorizer.transform(X, Y))

