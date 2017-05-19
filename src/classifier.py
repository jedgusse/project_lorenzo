#!/usr/bin/env

import json
from pprint import pprint
import os
import multiprocessing
import numpy as np
from scipy import stats
from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import (
    Normalizer, StandardScaler, FunctionTransformer)
from sklearn.metrics import precision_recall_fscore_support

from src.utils import docs_to_X, crop_docs
from src.data import DataReader


def to_dense(X):
    """
    Vectorizer outputs sparse matrix X
    This function returns X as a dense matrix
    """
    X = X.todense()
    return X


def deltavectorizer(X):
    """
    Function that normalizes X to Delta score
    "An expression of pure difference is what we need"
    - Burrows -> absolute Z-scores
    """
    X = stats.zscore(X)
    X = np.abs(X)
    # NaNs are replaced by zero
    X = np.nan_to_num(X)
    return X


def pipe_grid_clf(X_train, y_train):
    """
    Parameters
    ===========
    X_train : list of documents (where a document is a list of sents,
        where a sent is a str)
    y_train : list of transformed labels
    """
    # Initialize pipeline
    # Steps in the pipeline:
    # 1 Vectorize incoming training material
    # 2 Normalize counts
    # 3 Classify

    pipe = Pipeline(
        [('vectorizer', TfidfVectorizer()),
         ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
         ('classifier', svm.SVC())])

    # Classifier parameters

    idfs = [True, False]
    c_options = [1, 10, 100, 1000]
    kernel_options = ['linear', 'rbf']
    n_features_options = [1000, 3000, 5000, 10000, 15000, 30000]

    param_grid = [
        {
            'vectorizer': [TfidfVectorizer(), TfidfVectorizer(analyzer= 'char', ngram_range=(2,4))],
            'vectorizer__use_idf': idfs,
            'vectorizer__max_features': n_features_options,
            'vectorizer__norm': norm_options,
            'classifier__C': c_options,
            'classifier__kernel': kernel_options,
        },
    ]

    # Stratification is default
    # For integer/None inputs for cv, if the estimator is a classifier
    # and y is either binary or multiclass, StratifiedKFold is used.
    # In all other cases, KFold is used.
    n_jobs = multiprocessing.cpu_count()
    grid = GridSearchCV(pipe, cv=5, n_jobs=n_jobs, param_grid=param_grid, verbose=1)
    grid.fit(X_train, y_train)

    return grid


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Experiment directory to save results')
    parser.add_argument('--reader_path', help='Reader path', required=True)
    parser.add_argument('--generated_path', required=True,
                        help='Generated docs path')
    parser.add_argument('--max_words_train', default=False, type=int,
                        help='Number of words used per training/classify doc')
    args = parser.parse_args()

    # 1 Load alpha_bar documents
    X_alpha_bar, y_alpha_bar = [], []
    for fname in os.listdir(args.generated_path):
        author = fname.split('.')[0].replace('_', ' ')
        with open(os.path.join(args.generated_path, fname), 'r') as f:
            doc = [line.strip() for line in f]
        X_alpha_bar.append(doc), y_alpha_bar.append(author)
    assert len(X_alpha_bar), \
        "Couldn't find generated docs in %s" % args.generated_path
    gen_authors = set(y_alpha_bar)

    # 2 Load omega docs from reader
    reader = DataReader.load(args.reader_path)
    alpha, omega, _ = reader.foreground_splits()  # use gener split as test
    (y_alpha, _, X_alpha), (y_omega, _, X_omega) = alpha, omega
    # remove authors with no generator (because of missing docs)
    for idx, y in enumerate(y_alpha):
        if y not in gen_authors:
            del y_alpha[idx]
            del X_alpha[idx]
    for idx, y in enumerate(y_omega):
        if y not in gen_authors:
            del y_omega[idx]
            del X_omega[idx]
    # translate author names to labels
    le = preprocessing.LabelEncoder()
    le.fit(y_alpha)  # assumes that all three datasets have all authors

    # 3 Train estimator on alpha, omega and alpha_bar
    print("Training omega")
    if args.max_words_train:
        X_omega = crop_docs(X_omega, max_words=args.max_words_train)
    grid_omega = pipe_grid_clf(list(docs_to_X(X_omega)), le.transform(y_omega))
    print("Training alpha-bar")
    if args.max_words_train:
        X_alpha_bar = crop_docs(X_alpha_bar, max_words=args.max_words_train)
    grid_alpha_bar = \
        pipe_grid_clf(list(docs_to_X(X_alpha_bar)), le.transform(y_alpha_bar))
    print("Training alpha")
    if args.max_words_train:
        X_alpha = crop_docs(X_alpha, max_words=args.max_words_train)
    grid_alpha = pipe_grid_clf(list(docs_to_X(X_alpha)), le.transform(y_alpha))

    # 4 Test estimator on real and generated docs and save
    def run_test(grid, path, X_test, y_test, le):
        y_pred = grid.predict(docs_to_X(X_test))

        if not os.path.isdir(path):
            os.mkdir(path)

        def dump_report(y_true, y_pred, path, le):
            p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
            report = []
            for i in range(len(set(y_true))):
                report.append(
                    {'author': le.inverse_transform(i),
                     'result': {'precision': p[i],
                                'recall': r[i],
                                'f1': f1[i],
                                'support': int(s[i])}})
            with open(path, 'w') as f:
                json.dump(report, f)

        out_report_path = os.path.join(path, 'report.json')
        dump_report(le.transform(y_pred), y_test, out_report_path, le)
        with open(os.path.join(path, 'best_model.txt'), 'w') as f:
            pprint(grid.best_estimator_, stream=f)
        with open(os.path.join(path, 'best_params.json'), 'w') as f:
            json.dump({k: str(v) for k, v in grid.best_params_.items()}, f)

    omega_alpha = os.path.join(args.path, 'omega_alpha')
    run_test(grid_omega, omega_alpha, X_alpha, y_alpha, le)
    omega_alpha_bar = os.path.join(args.path, 'omega_alpha_bar')
    run_test(grid_omega, omega_alpha_bar, X_alpha_bar, y_alpha_bar, le)
    alpha_omega = os.path.join(args.path, 'alpha_omega')
    run_test(grid_alpha, alpha_omega, X_omega, y_omega, le)
    alpha_bar_omega = os.path.join(args.path, 'alpha_bar_omega')
    run_test(grid_alpha_bar, alpha_bar_omega, X_omega, y_omega, le)
