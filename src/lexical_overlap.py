
import os
import string
import argparse

from data import DataReader


class Ngrams:
    def __init__(self, n_list):
        self.n_list = n_list
        self.indices = {}

    def partial_fit(self, sentence):
        """Magic n-gram function fits to vector indices."""
        idx, inp = len(self.indices) - 1, sentence
        for n in self.n_list:
            for x in zip(*[inp[i:] for i in range(n)]):
                if self.indices.get(x) is None:
                    idx += 1
                    self.indices.update({x: idx})

    def fit(self, sentences):
        for s in sentences:
            self.partial_fit(s)
        return self

    def partial_transform(self, sentence):
        """Given a sentence, convert to a gram vector."""
        v, inp = [0] * len(self.indices), sentence
        for n in self.n_list:
            for x in zip(*[inp[i:] for i in range(n)]):
                if self.indices.get(x) is not None:
                    v[self.indices[x]] += 1
        return v

    def transform(self, sentences):
        for s in sentences:
            yield self.partial_transform(s)

    def fit_transform(self, sentences):
        return self.fit(sentences).transform(sentences)


def sents_from_file(path, max_words, processor):
    words, sents = 0, []
    with open(path, 'r') as f:
        for line in f:
            line = processor(line.strip())
            sents.append(line)
            words += len(line)
            if words >= max_words:
                return sents
    return sents


def sents_from_doc(doc, max_words, processor):
    words, sents = 0, []
    for sent in doc:
        sent = processor(sent)
        sents.append(sent)
        words += len(sent)
        if words >= max_words:
            return sents
    return sents


def processor(line):
    for ch in string.punctuation:
        line = line.replace(ch, '')
    return line.split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gener_path', help='dir with generated files')
    parser.add_argument('reader_path', help='path to saved reader')
    parser.add_argument('--max_words', type=int, default=50000)
    args = parser.parse_args()

    gener_ng = {}
    for f in os.listdir(args.gener_path):
        author = f.split('.')[0].replace('_', ' ')
        if author not in gener_ng:
            gener_ng[author] = Ngrams([2, 3, 4])
        fname = os.path.join(args.gener_path, f)
        gener_ng[author].fit(sents_from_file(fname, args.max_words, processor))

    reader = DataReader.load(args.reader_path)
    _, discrim, _ = reader.foreground_splits()
    discrim_authors, _, discrim_docs = discrim

    discrim_ng = {}
    for author, doc in zip(discrim_authors, discrim_docs):
        if author not in gener_ng:
            continue
        if author not in discrim_ng:
            discrim_ng[author] = Ngrams([2, 3, 4])
        discrim_ng[author].fit(sents_from_doc(doc, args.max_words, processor))
