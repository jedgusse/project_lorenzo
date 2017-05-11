#!/usr/bin/env

from pprint import pprint
import os
import argparse

from sklearn.metrics import classification_report
from sklearn import preprocessing

from data import DataReader
from generator import LMGenerator
from classifier import pipe_grid_clf, docs_to_X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--load_generated', action='store_true')
    parser.add_argument('--save_generated', action='store_true')                        
    parser.add_argument('--generated_path',
                        help='Path to generated files. Filename ' + \
                        'in format: authorname_docnum.txt')
    parser.add_argument('--nb_docs', default=10, type=int,
                        help='Number of generated docs per author')
    args = parser.parse_args()

    reader = DataReader.load(args.reader_path)
    gener, discrim, test = reader.foreground_splits()

    X_train, y_train, X_test, y_test = [], [], [], []

    # - Generate (or load) generated documents
    def generate_docs(generator, author, nb_docs, max_words,
                      save=False, path=None):
        print("::: Generating %d docs for %s" % (args.nb_docs, author))
        docs, scores = [], []
        max_sent_len = max(len(s) for s in generator.examples)
        for doc_id in range(nb_docs):
            doc, score = generator.generate_doc(
                max_words=max_words, max_sent_len=max_sent_len)
            docs.append(doc), scores.append(score)
            if save:
                doc_id = str(doc_id + 1).rjust(len(str(nb_docs)), '0')
                fname = '%s.%s.txt' % ('_'.join(author.split()), doc_id)
                with open(os.path.join(path, fname), 'w+') as f:
                    f.write('\n'.join(doc))
        return docs, author

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
            X_train.extend(docs), y_train.extend([author for _ in range(len(docs))])

    # - Load real data (for testing)
    # TODO

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
