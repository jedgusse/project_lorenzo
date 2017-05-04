#!/usr/bin/env

from preprocess import vectorize
from knn_metric import knn
from pca import principal_components_analysis
from hierarchy import dendrogram, heatmap, plot_frequencies, plot_loadings, gephi_networks
from vocabulary_richness import deviant_cwords
from rollingdelta import rolling_delta
from preprocess import vectorize, zscore_delta, tfidf_vectorizer
from sklearn.base import TransformerMixin
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, chi2, f_classif, mutual_info_classif, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, chi2, f_classif, mutual_info_classif, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from classifier import gridsearch_clf

# Sample length, amount of features, location of corpus folder

sample_length = 1500
feat_amount = 300
nearest_neighbors = 4
folder_location = "/Users/jedgusse/stylofactory/corpora/corpus_onlyfwords"
step_size = 300

invalid_words = ['dummyword']

# For classification tests, split data into training and test corpus (classifier will train and evaluate on training corpus,
# and predict on new test corpus)

test_train_split = 'yes'
test_dict = {'AnsLaon': 'Apocalypsim'}

# Main code

if __name__ == "__main__":

    authors, titles, texts, raw_counts, features = vectorize(folder_location, sample_length, feat_amount, invalid_words)
    zscore_vectors = zscore_delta(raw_counts, authors, titles, test_dict, test_train_split)
    tfidf_vectors = tfidf_vectorizer(raw_counts, authors, titles, features, test_dict, test_train_split)
    distances, indices = knn(tfidf_vectors, authors, titles, nearest_neighbors)
    grid_doc_vectors, used_features, best_feat_amount = gridsearch_clf(tfidf_vectors, authors, titles, features, test_dict, feat_amount)

    #principal_components_analysis(grid_doc_vectors, authors, titles, used_features, show_samples='yes', show_loadings='no')
    #dendrogram(zscore_vectors, authors, titles, features)
    #plot_loadings(authors, titles, raw_counts, zscore_vectors, features, feat_amount, "Bernard_vester_freq")
    #plot_frequencies(authors, titles, raw_counts, features, feat_amount, "Bernard_vester_freq")
    #heatmap(zscore_vectors, authors, titles, features)
    #gephi_networks(authors, titles, tfidf_vectors, nearest_neighbors)
    #deviant_cwords(folder_location)
    #rolling_delta(sample_length, feat_amount, invalid_words, step_size)

    # Write data to CSV file:

    """newcsv = open("/Users/jedgusse/stylofactory/output/text_output/data_exp1.csv", "w")

    newcsv.write("author, title, ")

    for feature in features:
    newcsv.write("{}, ".format(feature))
    newcsv.write("\n")

    for author, title, raw_vector in zip(authors, titles, raw_counts):
    newcsv.write(author + ", " + title + ", ")
    for number in raw_vector:
        newcsv.write(str(number) + ", ")
    newcsv.write("\n")"""


