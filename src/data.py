# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split

import readers

READERS = {'PL': readers.patrologia_reader,
           'PackHum': readers.packhum_reader}



class DataReader(object):

    def __init__(name='PL', foreground_authors=None):
        super(DataReader, self).__init__()
        self.name = name
        self.foreground_authors = foreground_authors
        self.reader = READERS(self.name)

    def background_sentences(self):
        for document in self.reader(exclude=foreground):
            for sentence in document.sentences:
                yield sentence

    def foreground_splits(self, discrim_size=.8):
        data = self.reader(include=foreground)

        authors = [d.author for d in data]
        titles = [d.title for d in data]
        sentences = [d.title for d in data]

        # first chop off discrim data:
        discrim_authors, rest_authors, \
            discrim_titles, rest_titles, \
            discrim_texts, rest_texts = train_test_split(
                authors, titles, texts,
                train_size=float(discrim_size),
                stratify=authors)

        # split remaining data in equally-sized gener and test
        gener_authors, test_authors, \
            gener_titles, test_titles, \
            gener_texts, test_texts = train_test_split(
                authors, titles, texts,
                train_size=int(len(rest_authors) / 2.0),
                stratify=rest_authors)

        return gener, discrim, test

reader = DataReader(name='PL')
