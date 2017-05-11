#!/usr/bin/env

from pprint import pprint
import os
import argparse
import multiprocessing

from joblib import Parallel, delayed

from sklearn.metrics import classification_report
from sklearn import preprocessing

from seqmod.utils import load_model

from utils import generate_docs, docs_to_X
from data import DataReader
from classifier import pipe_grid_clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--load_generated', action='store_true')
    parser.add_argument('--save_generated', action='store_true')
    parser.add_argument('--generated_path',
                        help='Path to generated files. Filename ' +
                        'in format: authorname_docnum.txt')
    parser.add_argument('--nb_docs', default=10, type=int,
                        help='Number of generated docs per author')
    args = parser.parse_args()

    # - Load reader and generator paths
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

    reader = DataReader.load(reader_path)
    gener, discrim, test = reader.foreground_splits()

    X_train, y_train, X_test, y_test = [], [], [], []

    # - Generate (or load) generated documents
    if args.load_generated:     # load generated documents
        assert not args.generated_path or os.path.isdir(args.generated_path), \
            "argument to --generated_path is not a valid path"
        for fname in os.listdir(args.generated_path):
            author = fname.split('.')[0].replace('_', ' ')
            doc = []
            with open(os.path.join(args.generated_path, fname), 'r') as f:
                for line in f:
                    doc.append(line.strip())
            X_train.append(doc), y_train.append(author)
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
            X_train.extend(docs)
            y_train.extend([author for _ in range(len(docs))])

    # - Load real data (for testing)
    _, discrim, _ = reader.foreground_splits()
    _, X_test, y_test = discrim

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    grid = pipe_grid_clf(X_train, y_train)

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
