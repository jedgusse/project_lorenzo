#!/usr/bin/env

import json
from pprint import pprint
import os
import multiprocessing
import numpy as np
from scipy import stats
from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import precision_recall_fscore_support

from src.utils import (
    docs_to_X, load_best_params, filter_authors, load_docs_from_dir)
from src.data import DataReader
from src.authors import authors


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
    kernel_options = ['linear']
    n_features_options = [5000, 10000, 15000, 30000]
    norm_options = ['l1', 'l2']

    param_grid = [
        {
            'vectorizer': [#TfidfVectorizer(),
                           TfidfVectorizer(analyzer='char',
                                           ngram_range=(2, 4))],
            #'vectorizer__use_idf': idfs,
            'vectorizer__max_features': n_features_options,
            #'vectorizer__norm': norm_options,
            'classifier__C': c_options,
            'classifier__kernel': kernel_options,
        },
    ]

    # Stratification is default
    # For integer/None inputs for cv, if the estimator is a classifier
    # and y is either binary or multiclass, StratifiedKFold is used.
    # In all other cases, KFold is used.
    n_jobs = multiprocessing.cpu_count()
    grid = GridSearchCV(
        pipe, cv=5, n_jobs=n_jobs, param_grid=param_grid, verbose=1)
    grid.fit(X_train, y_train)

    return grid


def clf_from_params(params):
    return Pipeline(
        [('vectorizer', TfidfVectorizer(
            analyzer='char',    # default to this
            ngram_range=(2, 4),  # default to this
            use_idf=params['vectorizer__use_idf'],
            max_features=params['vectorizer__max_features'],
            norm=params['vectorizer__norm'])),
         ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
         ('classifier', svm.SVC(
             C=params['classifier__C'],
             kernel=params['classifier__kernel']))])


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
    dump_report(le.transform(y_test), y_pred, out_report_path, le)
    if isinstance(grid, Pipeline):
        return              # only save best params if grid
    with open(os.path.join(path, 'best_model.txt'), 'w') as f:
        pprint(grid.best_estimator_, stream=f)
    with open(os.path.join(path, 'best_params.json'), 'w') as f:
        json.dump({k: str(v) for k, v in grid.best_params_.items()}, f)
    with open(os.path.join(path, 'cv_result.json'), 'w') as f:
        json.dump({k: str(v) for k, v in grid.cv_results_.items()}, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="To run omega_alpha, alpha_omega, pass reader_path (or both " +
        "omega_path and alpha_path), and omit alpha_bar_path. To run " +
        "alpha_bar_omega once you've run the first experiment, pass " +
        "alpha_bar_path, reader_path or omega_path and omega_params. " +
        "To run the self-learning experiments, also pass alpha_path")
    parser.add_argument('output_path', help='Directory to save results')
    parser.add_argument('--reader_path', help='Reader path')
    parser.add_argument('--omega_path', help='Omega docs path')
    parser.add_argument('--alpha_path', help='Omega docs path')
    parser.add_argument('--alpha_bar_path', help='Path to generated texts.' +
                        'If given, it will do the alpha_bar experiments.')
    parser.add_argument('--omega_params', help='Path to file containing the ' +
                        'already grid-searched params of the omega classifer')
    parser.add_argument('--max_authors', type=int, default=50, help='Run ' +
                        'experiments for a selection of the max n authors')
    args = parser.parse_args()

    # 1 Load documents
    X_alpha, y_alpha = None, None
    X_omega, y_omega = None, None
    X_alpha_bar, y_alpha_bar = None, None

    if args.reader_path:
        reader = DataReader.load(args.reader_path)
        alpha, omega, _ = reader.foreground_splits()  # use gener split as test
        (y_alpha, _, X_alpha), (y_omega, _, X_omega) = alpha, omega
    if args.alpha_path:
        X_alpha, y_alpha = load_docs_from_dir(args.alpha_path)
    if args.omega_path:
        X_omega, y_omega = load_docs_from_dir(args.omega_path)
    if args.alpha_bar_path:
        X_alpha_bar, y_alpha_bar = load_docs_from_dir(args.alpha_bar_path)

    # 2 Eventually filter authors
    keep_authors = set(
        [a for a in y_alpha_bar if a in authors[:args.max_authors]])
    if X_alpha is not None:
        y_alpha, X_alpha = filter_authors(y_alpha, X_alpha, keep_authors)
    if X_omega is not None:
        y_omega, X_omega = filter_authors(y_omega, X_omega, keep_authors)
    if X_alpha_bar is not None:
        y_alpha_bar, X_alpha_bar = filter_authors(
            y_alpha_bar, X_alpha_bar, keep_authors)
    # translate author names to labels
    le = preprocessing.LabelEncoder()
    le.fit([author for y in [y_alpha, y_omega, y_alpha_bar] for author in y])
    print("Fitted %d classes" % len(le.classes_))
    pprint({a: idx for a, idx in zip(le.classes_, le.transform(le.classes_))})

    # 3 Train estimator on alpha, omega and alpha_bar
    # 3.1 Train omega
    if X_omega is not None:
        if args.omega_params is not None:
            print("Loading omega best params")
            grid_omega = clf_from_params(load_best_params(args.omega_params))
            grid_omega.fit(docs_to_X(X_omega), le.transform(y_omega))
        else:
            print("Training omega")
            grid_omega = pipe_grid_clf(docs_to_X(X_omega), le.transform(y_omega))
            # classify alpha only if not loaded
            omega_alpha_path = os.path.join(args.output_path, 'omega_alpha')
            run_test(grid_omega, omega_alpha_path, X_alpha, y_alpha, le)
    if X_alpha_bar is not None:
        print("Classifying alpha bar")
        omega_alpha_bar_path = os.path.join(args.output_path, 'omega_alpha_bar')
        run_test(grid_omega, omega_alpha_bar_path, X_alpha_bar, y_alpha_bar, le)

    # 3.2 Train alpha
    if X_alpha is not None and X_alpha_bar is None:
        assert X_omega, "Need omega docs to perform alpha_omega experiments"
        print("Training alpha")
        grid_alpha = pipe_grid_clf(docs_to_X(X_alpha), le.transform(y_alpha))
        # classify omega
        print("Classifying omega")
        alpha_omega_path = os.path.join(args.output_path, 'alpha_omega')
        run_test(grid_alpha, alpha_omega_path, X_omega, y_omega, le)

    # 3.3 Train alpha_bar
    if X_alpha_bar is not None:
        assert X_omega, "Need omega docs to perform X_alpha_bar experiments"
        print("Training alpha-bar")
        grid_alpha_bar = \
            pipe_grid_clf(docs_to_X(X_alpha_bar), le.transform(y_alpha_bar))
        print("Classifying omega")
        alpha_bar_omega_path = os.path.join(args.output_path, 'alpha_bar_omega')
        run_test(grid_alpha_bar, alpha_bar_omega_path, X_omega, y_omega, le)

        # 3.4 Self-training experiment
        if X_alpha is not None:
            print("Training alpha+alpha-bar")
            grid_alpha_alpha_bar = \
                pipe_grid_clf(docs_to_X(X_alpha_bar + X_alpha),
                              le.transform(y_alpha + y_alpha_bar))
            print("Classifying omega")
            alpha_alpha_bar_omega_path = os.path.join(
                args.output_path, 'alpha_alpha_bar_omega')
            run_test(grid_alpha_alpha_bar, alpha_alpha_bar_omega_path,
                     X_omega, y_omega, le)
