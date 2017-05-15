#!/usr/bin/env

import json
from pprint import pprint
import os

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
         ('feature_scaling', Normalizer()),
         ('classifier', svm.SVC())])

    # GridSearch parameters
    # Diminish feature dimensionality to feature amount of N:
    n_features_options = list(range(20, 600, 20))

    # Classifier parameters
    # C parameter: optimize towards smaller-margin hyperplane (large C)
    # or larger-margin hyperplane (small C)
    # Therefore C is the penalty parameter of the error term.
    c_options = [1, 10, 100, 1000]
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    analyzer_options = ['char_wb']
    ngram_range_options = [(2,2), (3,3), (4,4)]

    param_grid = [
        {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'feature_scaling': [StandardScaler(),
                                Normalizer(),
                                FunctionTransformer(deltavectorizer)],
            'classifier__C': c_options,
            'classifier__kernel': kernel_options,
        },
        {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
            'vectorizer__analyzer': analyzer_options,
            'vectorizer__ngram_range': ngram_range_options,
            'feature_scaling': [StandardScaler(),
                                Normalizer(),
                                FunctionTransformer(deltavectorizer)],
            'classifier__C': c_options,
            'classifier__kernel': kernel_options,
        },
    ]
    
    # Stratification is default
    # For integer/None inputs for cv, if the estimator is a classifier
    # and y is either binary or multiclass, StratifiedKFold is used.
    # In all other cases, KFold is used.
    grid = GridSearchCV(pipe, cv=LeaveOneOut(), n_jobs=2, param_grid=param_grid, verbose=1)
    grid.fit(X_train, y_train)

    return grid


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Top directory containing reader, ' +
                        'generators, and generated docs as per generator.py')
    parser.add_argument('--reader_path', help='Custom reader path')
    parser.add_argument('--generated_path', help='Custom generated docs path')
    parser.add_argument('--max_words_train', default=False, type=int,
                        help='Number of words used per training/classify doc')
    args = parser.parse_args()

    # 1 Load generated documents
    X_gen, y_gen = [], []
    if args.generated_path is not None:
        generated_path = args.generated_path
    else:                       # use default generated path
        generated_path = os.path.join(args.path, 'generated')
    for fname in os.listdir(generated_path):
        author = fname.split('.')[0].replace('_', ' ')
        with open(os.path.join(generated_path, fname), 'r') as f:
            doc = [line.strip() for line in f]
        X_gen.append(doc), y_gen.append(author)
    assert len(X_gen), "Couldn't find generated docs in %s" % generated_path
    gen_authors = set(y_gen)

    # 2 Load real docs from reader
    reader_path = None
    if args.reader_path:        # load reader from custom path
        reader_path = args.reader_path
    for f in os.listdir(args.path):
        if f.endswith('pkl'):   # reader path
            reader_path = os.path.join(args.path, f)
    if not reader_path:
        raise ValueError("Couldn't find reader in dir [%s]" % args.path)
    reader = DataReader.load(reader_path)
    test, train, _ = reader.foreground_splits()  # use gener split as test
    (y_train, _, X_train), (y_test, _, X_test) = train, test
    if args.max_words_train:
        X_train = list(crop_docs(X_train, max_words=args.max_words_train))
    # remove authors with no generator (because of missing docs)
    for idx, y in enumerate(y_train):
        if y not in gen_authors:
            del y_train[idx]
            del X_train[idx]
    for idx, y in enumerate(y_test):
        if y not in gen_authors:
            del y_test[idx]
            del X_test[idx]
    # translate author names to labels
    le = preprocessing.LabelEncoder()
    le.fit(y_train)

    # 3 Train estimator on real data and generated data
    grid_real = pipe_grid_clf(docs_to_X(X_gen), le.transform(y_gen))
    grid_gen = pipe_grid_clf(docs_to_X(X_train), le.transform(y_train))

    # 4 Test estimator on real and generated docs and save
    def run_test(grid, path, X_in, y_in, X_out, y_out, le):
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        in_pred = grid.predict(docs_to_X(X_in))
        out_pred = grid.predict(docs_to_X(X_out))

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

        out_report_path = os.path.join(path, 'report_out.json')
        dump_report(le.transform(y_out), out_pred, out_report_path, le)
        in_report_path = os.path.join(path, 'report_in.json')
        dump_report(le.transform(y_in), in_pred, in_report_path, le)
        with open(os.path.join(path, 'best_model.txt'), 'w') as f:
            pprint(best_model, stream=f)
        with open(os.path.join(path, 'best_params.json'), 'w') as f:
            json.dump({k: str(v) for k, v in best_params.items()}, f)

    class_path = os.path.join(args.path, 'classification')
    run_test(grid_real, class_path, X_test, y_test, X_gen, y_gen, le)
    gen_path = os.path.join(args.path, 'augmentation')
    run_test(grid_gen, gen_path, X_gen, y_gen, X_test, y_test, le)
