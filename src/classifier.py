#!/usr/bin/env

from pprint import pprint
import os
from joblib import Parallel, delayed
import multiprocessing

import numpy as np
from scipy import stats
from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import (
    Normalizer, StandardScaler, FunctionTransformer)
from sklearn.metrics import classification_report

from seqmod.utils import load_model

from utils import generate_docs, docs_to_X
from generator import LMGenerator
from data import DataReader


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

    param_grid = [
        {
            'vectorizer': [CountVectorizer(), TfidfVectorizer()],
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
    grid = GridSearchCV(pipe, cv=2, n_jobs=2, param_grid=param_grid, verbose=1)
    grid.fit(X_train, y_train)

    return grid


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Top directory containing reader ' +
                        'and generators as per generator.py')
    parser.add_argument('--generated_path',
                        help='Path to generated files. Filename in format: ' +
                        'authorname_docnum.txt. It will be used as target ' +
                        'write dir if save_generated is passed or as target ' +
                        'read dir if load_generated is passed.')
    parser.add_argument('--reader_path', help='Custom reader path.')
    parser.add_argument('--save_generated', action='store_true',
                        help='Whether to save the generated texts')
    parser.add_argument('--load_generated', action='store_true')
    parser.add_argument('--nb_docs', default=10, type=int,
                        help='Number of generated docs per author')
    parser.add_argument('--max_words', default=2000, type=int,
                        help='Number of words per generated doc')
    args = parser.parse_args()
    assert not (args.save_generated and not args.generated_path), \
        "--save_generated requires --generated_path"
    assert not (args.load_generated and not args.generated_path), \
        "--load_generated requires --generated_path"

    # 1 Load discriminator data and generator models/data
    reader_path, generators = None, {}
    for f in os.listdir(args.path):
        if f.endswith('pkl'):   # reader path
            reader_path = os.path.join(args.path, f)
        elif f.endswith('pt'):  # generator path
            basename = os.path.splitext(os.path.basename(f))[0]
            author = ' '.join(basename.split('_'))
            generators[author] = os.path.join(args.path, f)

    if args.reader_path:        # load reader from custom path
        reader_path = args.reader_path

    if not reader_path:
        raise ValueError("Couldn't find reader in dir [%s]" % args.path)

    # generate (or load) generated documents
    X_gen, y_gen = [], []

    if args.load_generated:     # load generated documents
        assert not args.generated_path or os.path.isdir(args.generated_path), \
            "argument to --generated_path is not a valid path"
        for fname in os.listdir(args.generated_path):
            author = fname.split('.')[0].replace('_', ' ')
            doc = []
            with open(os.path.join(args.generated_path, fname), 'r') as f:
                for line in f:
                    doc.append(line.strip())
            X_gen.append(doc), y_gen.append(author)
    else:                       # generate documents
        if args.save_generated and not os.path.isdir(args.generated_path):
            os.mkdir(args.generated_path)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
            delayed(generate_docs)(
                load_model(fpath), author, args.nb_docs, args.max_words,
                save=args.save_generated, path=args.generated_path)
            for author, fpath in generators.items())
        for docs, author in results:
            X_gen.extend(docs)
            y_gen.extend([author for __ in range(len(docs))])
        """
        # Single-threaded code
        for author, fpath in generators.items():
            generator = load_model(fpath)
            docs, _ = generate_docs(
                generator, author, args.nb_docs, args.max_words,
                save=args.save_generated, path=args.generated_path)
            X_gen.extend(docs)
            y_gen.extend([author for _ in range(len(docs))])
        """

    # 2 Compute best estimator on real data
    reader = DataReader.load(reader_path)
    test, train, _ = reader.foreground_splits()  # use gener split as test
    (y_train, _, X_train), (y_test, _, X_test) = train, test

    # remove authors with no generator (because training failed)
    for idx, y in enumerate(y_train):
        if y not in generators:
            del y_train[idx]
            del X_train[idx]
    for idx, y in enumerate(y_test):
        if y not in generators:
            del y_test[idx]
            del X_test[idx]

    # translate author names to labels
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    print("::: Encoded labels :::")
    labels = list(generators.keys())
    idxs = le.inverse_transform(labels)
    print('\n'.join(['%s: %d' % (l, idx) for l, idx in zip(labels, idxs)]))
    grid = pipe_grid_clf(docs_to_X(X_train), y_train)

    # make prediction with the best parameters
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    accuracy = grid.best_score_ * 100
    prediction = grid.predict(docs_to_X(X_test))

    print("::: Best model :::")
    pprint(best_model)
    print("::: Best model params :::")
    pprint(best_params)
    print("::: CV Accuracy :::", "%g" % accuracy)
    print("::: Classification report :::")
    print(classification_report(le.transform(y_test), prediction))

    # 3 Test estimator on generated docs
    gen_pred = grid.predict(docs_to_X(X_gen))
    print("::: Generation classification report :::")
    print(classification_report(le.transform(y_gen), gen_pred))
