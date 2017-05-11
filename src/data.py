# -*- coding: utf-8 -*-

import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

from sklearn.model_selection import train_test_split

import readers

READERS = {'PL': readers.patrologia_reader,
           'PackHum': readers.packhum_reader}


class DataReader(object):
    def __init__(self, name='PL', foreground_authors=None):
        super(DataReader, self).__init__()
        self.name = name
        self.foreground_authors = foreground_authors
        self.reader = READERS[self.name]

    def background_sentences(self):
        for document in self.reader(exclude=self.foreground_authors):
            for sentence in document.sentences:
                yield sentence

    def foreground_splits(self, gener_size=.4, discrim_size=.6):
        """
        gener_size : float, proportion of text allocated to generator training
            (with respect to the total data)
        discrim_size : float, proportion of text allocated to discriminator
            training (after splitting generation data). This is the training-
            test split proportion after generation split.
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
                stratify=authors)

        # split remaining data in equally-sized discrim and test
        discrim_authors, test_authors, \
            discrim_titles, test_titles, \
            discrim_texts, test_texts = train_test_split(
                rest_authors, rest_titles, rest_texts,
                train_size=float(discrim_size),
                stratify=rest_authors)

        return (gener_authors, gener_titles, gener_texts), \
            (discrim_authors, discrim_titles, discrim_texts), \
            (test_authors, test_titles, test_texts)

    def save(self, path, gener_size=.4, discrim_size=.6):
        authors = ['-'.join(a.replace('.', '').split())
                   for a in self.foreground_authors]
        fp = path + '{name}.{foreground_authors}.pkl'.format(
            name=self.name,
            foreground_authors='_'.join(authors))
        with open(fp, 'wb') as f:
            splits = self.foreground_splits(gener_size=gener_size,
                                            discrim_size=discrim_size)
            pickle.dump({'splits': splits,
                         'gener_size': gener_size,
                         'discrim_size': discrim_size,
                         'foreground_authors': self.foreground_authors,
                         'name': self.name},
                        f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        class LoadedReader(DataReader):
            def foreground_splits(self, discrim_size=None, gener_size=None):
                if discrim_size is not None or gener_size is not None:
                    logging.warn("Loaded reader doesn't allow changing " +
                                 "the foreground split proportions, " +
                                 "`(discrim|gener)_size` will be ignored. " +
                                 "gener_size and discrim_size are %g, %g" %
                                 (obj['gener_size'], obj['discrim_size']))
                return obj['splits']

        return LoadedReader(name=obj['name'],
                            foreground_authors=obj['foreground_authors'])


if __name__ == '__main__':
    foreground_authors = ('Tertullianus', 'Hieronymus Stridonensis')
    reader = DataReader(name='PL', foreground_authors=foreground_authors)
    reader.save('./')
    reader = DataReader.load('PL.Tertullianus_Hieronymus-Stridonensis.pkl')
    reader.foreground_splits()
