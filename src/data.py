# -*- coding: utf-8 -*-

import sklearn
import readers
if int(sklearn.__version__.split('.')[1]) < 18:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

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

        return (gener_authors, gener_texts, gener_titles), \
            (discrim_authors, discrim_texts, discrim_titles), \
            (test_authors, test_texts, test_titles)


if __name__ == '__main__':
    foreground_authors = ('Tertullianus', 'Hieronymus Stridonensis')
    reader = DataReader(name='PL', foreground_authors=foreground_authors)
    gen, disc, test, = reader.foreground_splits()
    authors, sents, titles = disc
