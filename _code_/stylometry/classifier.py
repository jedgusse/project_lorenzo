#!/usr/bin/env

from preprocess import vectorize, zscore_delta, tfidf_vectorizer
import numpy as np
import matplotlib.pyplot as plt
from word_counter import mfw_counter
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from itertools import compress

def gridsearch_clf(doc_vectors, authors, titles, features, test_dict, feat_amount):

	# Split data into training and test corpus

	le = preprocessing.LabelEncoder()
	labels = le.fit_transform(authors)

	test_authors = []
	test_titles = []

	# Instantiate test corpus
	x_test = []
	y_test = []

	# Instantiate training corpus
	X_train = []
	Y_train = []

	for doc_vector, author, title, label in zip(doc_vectors, authors, titles, labels):
		if author in test_dict and title.split("_")[0] in test_dict.values():
			x_test.append(doc_vector)
			y_test.append(label)
			test_authors.append(author)
			test_titles.append(title)
		else:
			X_train.append(doc_vector)
			Y_train.append(label)

	# Transformation of list object into array
	# Test corpus

	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)

	# Training corpus

	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)

	# First we predefine a pipeline that compiles two consecutive models (or steps) in the model:
	
	# 1) Dimensionality reduction
	# 2) Classification
	
	# The aim is to find the best possible parameters for both operations in a single CV run.

	# Predefine pipeline with two consecutive steps
	# The first step is dimensionality reduction, the second is classification

	pipe = Pipeline([('reduce_dim', PCA()), ('classify', svm.SVC())])

	n_features_options = list(range(20, 200, 20))
	c_options = [1, 10, 100, 1000]
	kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']

	param_grid = [
	    {
	        'reduce_dim': [SelectKBest(chi2), SelectKBest(f_regression)],
	        'reduce_dim__k': n_features_options,
	        'classify__C': c_options,
	        'classify__kernel': kernel_options,
	    },
	]

	# Stratification is default
	# For integer/None inputs for cv, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. 
	# In all other cases, KFold is used.s

	grid = GridSearchCV(pipe, cv=5, n_jobs=2, param_grid=param_grid)
	grid.fit(X_train, Y_train)

	# Make prediction with the best parameters

	best_feat_amount = grid.best_params_['reduce_dim'].get_params()['k']
	features_booleans = grid.best_params_['reduce_dim'].get_support()
	used_features = list(compress(features, features_booleans))

	best_model = grid.best_estimator_
	prediction = grid.predict(x_test)

	grid_doc_vectors = []
	for doc_vector in doc_vectors:
		compressed_doc_vec = list(compress(doc_vector, features_booleans))
		grid_doc_vectors.append(compressed_doc_vec)
	grid_doc_vectors = np.array(grid_doc_vectors)

	# Terminal output

	print()
	print("-- BEST MODEL -- | ", best_model)
	print()
	print("-- SCORE -- | ", "{}%".format(str(grid.best_score_*100)))
	print("-- AMOUNT OF FEATURES -- | ", "{}".format(str(best_feat_amount)))
	print("-- CHOSEN FEATURES -- | ", used_features)
	print()
	print("-- PREDICTIONS -- | ")

	for prediction, title in zip(le.inverse_transform(prediction), test_titles):
		print(prediction, title)

	print()

	# Returns

	return grid_doc_vectors, used_features, best_feat_amount


