# -*- coding: utf-8 -*-

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

    def foreground_splits(self, discrim_size=.8):
        data = list(self.reader(include=self.foreground_authors))

        authors = [d.author for d in data]
        titles = [d.title for d in data]
        sentences = [d.sentences for d in data]

        # first chop off discrim data:
        discrim_authors, rest_authors, \
            discrim_titles, rest_titles, \
            discrim_texts, rest_texts = train_test_split(
                authors, titles, sentences,
                train_size=float(discrim_size),
                stratify=authors)

        # split remaining data in equally-sized gener and test
        gener_authors, test_authors, \
            gener_titles, test_titles, \
            gener_texts, test_texts = train_test_split(
                rest_authors, rest_titles, rest_texts,
                train_size=int(len(rest_authors) / 2.0),
                stratify=rest_authors)

        return (gener_authors, gener_titles, gener_texts), \
            (discrim_authors, discrim_titles, discrim_texts), \
            (test_authors, test_titles, test_texts)

    def save(self, path, foreground_splits=.8):
        authors = ['-'.join(a.replace('.', '').split())
                   for a in self.foreground_authors]
        fp = path + '{name}.{foreground_authors}.pkl'.format(
            name=self.name,
            foreground_authors='_'.join(authors))
        with open(fp, 'wb') as f:
            pickle.dump({'splits': self.foreground_splits(),
                         'foreground_authors': self.foreground_authors,
                         'name': self.name},
                        f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        class LoadedReader(DataReader):
            def foreground_splits(self, foreground_splits=None):
                if foreground_splits is not None:
                    print("Loaded reader doesn't allow changing " +
                          "the foreground split proportions")
                return obj['splits']

        return LoadedReader(name=obj['name'],
                            foreground_authors=obj['foreground_authors'])


if __name__ == '__main__':
    foreground_authors = ('Tertullianus', 'Hieronymus Stridonensis')
    reader = DataReader(name='PL', foreground_authors=foreground_authors)
    reader.save('./')
    reader = DataReader.load('PL.Tertullianus_Hieronymus-Stridonensis.pkl')
