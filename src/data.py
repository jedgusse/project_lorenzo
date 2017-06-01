#!/usr/bin/env

import random; random.seed(1000)
from collections import defaultdict
import os
import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

from sklearn.model_selection import train_test_split

from src import readers

READERS = {'PL': readers.patrologia_reader,
           'PackHum': readers.packhum_reader}


class DataReader(object):
    def __init__(self, name='PL', foreground_authors=None, seed=1000):
        super(DataReader, self).__init__()
        self.name = name
        self.foreground_authors = foreground_authors
        self.reader = READERS[self.name]
        self.seed = seed

    def background_sentences(self):
        for document in self.reader(exclude=self.foreground_authors):
            for sentence in document.sentences:
                yield sentence

    def foreground_splits(self, gener_size=.5, discrim_size=.5, test=False):
        """
        gener_size : float, proportion of text allocated to generator training
            (with respect to the total data)
        discrim_size : float, proportion of text allocated to discriminator
            training (after splitting generation data). This is the training-
            test split proportion after generation split.
        test : bool, whether to use a third split for testing. If False,
            `discrim_size` will be set to 1 - gener_size.
        """
        data = list(self.reader(include=self.foreground_authors))

        authors = [d.author for d in data]
        titles = [d.title for d in data]
        sentences = [d.sentences for d in data]

        # first chop off gener data:
        gener_authors, rest_authors, \
            gener_titles, rest_titles, \
            gener_texts, rest_texts = train_test_split(
                authors, titles, sentences,
                train_size=float(gener_size),
                stratify=authors,
                random_state=self.seed)

        if test:
            # split remaining data in equally-sized discrim and test
            discrim_authors, test_authors, \
                discrim_titles, test_titles, \
                discrim_texts, test_texts = train_test_split(
                    rest_authors, rest_titles, rest_texts,
                    train_size=float(discrim_size),
                    stratify=rest_authors,
                    random_state=self.seed)
            return (gener_authors, gener_titles, gener_texts), \
                (discrim_authors, discrim_titles, discrim_texts), \
                (test_authors, test_titles, test_texts)

        else:
            # return test split as discrim split
            return (gener_authors, gener_titles, gener_texts), \
                (rest_authors, rest_titles, rest_texts), None

    def save(self, path, gener_size=.5, discrim_size=.5, **kwargs):
        fname = path if path.endswith('.pkl') else path + '.pkl'
        try:
            with open(fname, 'wb') as f:
                splits = self.foreground_splits(
                    gener_size=gener_size, discrim_size=discrim_size, **kwargs)
                reader_data = {'splits': splits,
                               'seed': self.seed,
                               'foreground_authors': self.foreground_authors,
                               'name': self.name}
                reader_data.update(kwargs)
                pickle.dump(reader_data, f)
        except AssertionError as e:
            os.remove(fname)
            raise e

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        class LoadedReader(DataReader):
            def foreground_splits(self, **kwargs):
                if kwargs:
                    logging.warn("Loaded reader doesn't allow changing " +
                                 "split params and params will be ignored." +
                                 "Current values are " +
                                 '\n'.join(['{}: {}'.format(k, v)
                                            for k, v in obj.items()
                                            if k not in ('splits', 'name')]))
                return obj['splits']

        args = {'name': obj['name'],
                'foreground_authors': obj['foreground_authors']}
        if 'seed' in obj:
            args['seed'] = obj['seed']

        return LoadedReader(**args)


def sample_split(docs, authors, nb_docs, nb_words):
    sents, sample = defaultdict(list), defaultdict(list)
    for doc, author in zip(docs, authors):
        for s in doc:
            sents[author].append(s)
    wcs = defaultdict(int)
    for author, sents in sents.items():
        random.shuffle(sents)
        doc = []
        for s in sents:
            if len(sample[author]) == nb_docs:
                break
            if wcs[author] >= nb_words:
                sample[author].append(doc)
                doc = []
                wcs[author] = 0
            doc.append(s)
            wcs[author] += len(s.strip().split())
    return sample


def dump_sample(sampled, path):
    for author, docs in sampled.items():
        for doc_id, doc in enumerate(docs):
            doc_id = str(doc_id + 1).rjust(len(str(len(docs))), '0')
            fname = '%s.%s.txt' % ('_'.join(author.split()), doc_id)
            with open(os.path.join(path, fname), 'w+') as f:
                f.write('\n'.join(doc))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--foreground_authors',
                        default=('Augustinus Hipponensis',
                                 'Hieronymus Stridonensis',
                                 'Bernardus Claraevallensis',
                                 'Walafridus Strabo'),
                        type=lambda args: args.split(','))
    parser.add_argument('--gener_size', default=.5, type=float)
    parser.add_argument('--discrim_size', default=.5, type=float)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--path', default='test')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--omega_path', help='Output path for sampled docs ' +
                        'from discriminator training data')
    parser.add_argument('--alpha_path', help='Output path for sampled docs ' +
                        'from generator training data')
    parser.add_argument('--nb_docs', type=int, default=20)
    parser.add_argument('--nb_words', type=int, default=5000)
    args = parser.parse_args()

    reader = DataReader(name='PL', foreground_authors=args.foreground_authors,
                        seed=args.seed)
    reader.save(args.path, gener_size=args.gener_size,
                discrim_size=args.discrim_size, test=args.test)
    alpha, omega, _ = reader.foreground_splits()  # use gener split as test
    (y_alpha, _, X_alpha), (y_omega, _, X_omega) = alpha, omega
    if args.omega_path:
        omega = sample_split(X_omega, y_omega, args.nb_docs, args.nb_words)
        dump_sample(omega, args.omega_path)
    if args.alpha_path:
        alpha = sample_split(X_alpha, y_alpha, args.nb_docs, args.nb_words)
        dump_sample(alpha, args.alpha_path)
