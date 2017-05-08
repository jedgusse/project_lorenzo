#!/usr/bin/env

from itertools import compress
import numpy as np
from scipy import stats
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.base import TransformerMixin
from sklearn import preprocessing

def to_dense(X):
        # Vectorizer outputs sparse matrix X
        # This function returns X as a dense matrix
    X = X.todense()
    return X

def deltavectorizer(X):
        # Function that normalizes X to Delta score
        # "An expression of pure difference is what we need" - Burrows -> absolute Z-scores
        # NaNs are replaced by zero
    X = stats.zscore(X)
    X = np.abs(X)
    X = np.nan_to_num(X)
    return X

def pipe_grid_clf(X_train, Y_train, x_test):
    
    # Translate author names to labels

    le = preprocessing.LabelEncoder()
    Y_train = le.fit_transform(Y_train)

    # Initialize pipeline
        # Steps in the pipeline:
        # 1 Vectorize incoming training material
        # 2 Normalize counts
        # 3 Classify

    pipe = Pipeline([('vectorizer', TfidfVectorizer()), ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)), 
                     ('feature_scaling', Normalizer()), ('classifier', svm.SVC())])

    # GridSearch parameters
        # Diminish feature dimensionality to feature amount of N:

    n_features_options = list(range(20, 600, 20))

        # Classifier parameters
        # C parameter: optimize towards smaller-margin hyperplane (large C) or larger-margin hyperplane (small C)
        # Therefore C is the penalty parameter of the error term.
    c_options = [1, 10, 100, 1000]
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']

    param_grid = [
    {
        'vectorizer': [CountVectorizer(), TfidfVectorizer()],
        'feature_scaling': [StandardScaler(), Normalizer(), FunctionTransformer(deltavectorizer)],
        'classifier__C': c_options,
        'classifier__kernel': kernel_options,
    },
    ]

    # Stratification is default
    # For integer/None inputs for cv, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used.
    # In all other cases, KFold is used.

    grid = GridSearchCV(pipe, cv=2, n_jobs=2, param_grid=param_grid, verbose=1)
    grid.fit(X_train, Y_train)

    # Make prediction with the best parameters

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    accuracy = grid.best_score_*100
    prediction = grid.predict(x_test)

    # Terminal Results

    print()
    print("::: Best model :::") 
    print()
    print(best_model)
    print()
    print(best_params)
    print()
    print("::: Score :::", "{} %".format(str(accuracy)))
    print()
    print("Authors :", le.inverse_transform(prediction))


