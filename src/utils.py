#!/usr/bin/env

import os
import string
import json
import numpy as np

from src.data import DataReader


def docs_to_X(docs):
    """
    Joins sentences in a collection of documents
    """
    return [' '.join(doc) for doc in docs]


def crop_doc(doc, max_words):
    """
    Reduce doc length to a maximum of `max_words`
    """
    words, sents = 0, []
    for sent in doc:
        words += len(sent.split())
        sents.append(sent)
        if words >= max_words:
            break
    return sents


def crop_docs(docs, max_words=float('inf')):
    """
    Run `crop_doc` on all docs
    """
    return [crop_doc(doc, max_words) for doc in docs]


def filter_authors(y_docs, X_docs, authors):
    y_docs, X_docs = zip(*[(y, x) for y, x in zip(y_docs, X_docs) if y in authors])
    return list(y_docs), list(X_docs)


def sample(a, temperature=1.0):
    """
    numpy implementation of multinomial sampling
    """
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def generate_docs(generator, author, nb_docs, max_words,
                  save=False, path=None):
    """
    Utility function to generate docs and save them
    """
    print("::: Generating %d docs for %s" % (nb_docs, author))
    if hasattr(generator, "cpu"):  # move to cpu if needed
        generator.cpu()
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


def train_generator(generator, author, examples, fitted_d, args):
    """
    Utility function to train a generator
    """
    generator.fit(
        examples, fitted_d, args.batch_size, args.bptt,
        args.epochs, gpu=args.gpu, add_hook=args.add_hook)
    fpath = '%s/%s' % (args.save_path, '_'.join(author.split()))
    suffix = '.pkl'
    if args.model == 'rnn_lm':
        generator.eval()        # set to validation mode
        suffix = '.pt'
    generator.save(fpath)
    return fpath + suffix


# loaders
def processor(line):
    for ch in string.punctuation:
        line = line.replace(ch, '')
    return line.split()


def sents_from_iter(iterator, max_words, processor=processor):
    words, sents = 0, []
    for sent in iterator:
        sent = processor(sent)
        sents.append(sent)
        words += len(sent)
        if words >= max_words:
            return sents
    return sents


def gener_sents(author, path):
    for f in os.listdir(path):
        f_author = f.split('.')[0].replace('_', ' ')
        if f_author == author:
            with open(os.path.join(path, f), 'r') as inf:
                for l in inf:
                    yield l.strip()


def real_sents(authors, docs, author):
    for a, doc in zip(authors, docs):
        if a == author:
            for sent in doc:
                yield sent


def load_gener_authors(path, max_words=1000000):
    authors_set = set(f.split('.')[0].replace('_', ' ')
                      for f in os.listdir(path))
    gener_ng = {}
    for author in authors_set:
        gener_ng[author] = sents_from_iter(gener_sents(author, path),
                                           max_words=max_words)
    return gener_ng


def load_real_authors(path, max_words=1000000):
    reader = DataReader.load(path)
    gener, discrim, _ = reader.foreground_splits()
    discrim_authors, _, discrim_docs = discrim
    gener_authors, _, gener_docs = gener

    discrim_ng = {}
    for author in set(discrim_authors):
        sents = sents_from_iter(
            real_sents(discrim_authors, discrim_docs, author),
            max_words=max_words)
        discrim_ng[author] = sents

    gener_source_ng = {}
    for author in set(discrim_authors):
        sents = sents_from_iter(
            real_sents(gener_authors, gener_docs, author),
            max_words=max_words)
        gener_source_ng[author] = sents

    return discrim_ng, gener_source_ng


def load_best_params(path):
    with open(os.path.join(path, 'best_params.json'), 'r') as f:
        params = json.load(f)
    if 'vectorizer__use_idf' in params:
        params['vectorizer__use_idf'] = \
            True if params['vectorizer__use_idf'] == 'True' else False
        params['vectorizer__max_features'] = \
            int(params['vectorizer__max_features'])
        params['classifier__C'] = int(params['classifier__C'])
    return params
