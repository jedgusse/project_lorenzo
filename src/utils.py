
import os
import numpy as np


def generate_docs(generator, author, nb_docs, max_words,
                  save=False, path=None):
    print("::: Generating %d docs for %s" % (nb_docs, author))
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


def docs_to_X(docs):
    """
    Joins sentences in a collection of documents
    """
    return [' '.join(doc) for doc in docs]


def crop_doc(doc, max_words):
    words, sents = 0, []
    for sent in doc:
        words += len(sent.split())
        sents.append(sent)
        if words >= max_words:
            break
    return sents


def crop_docs(docs, max_words=float('inf')):
    for doc in docs:
        yield crop_doc(doc, max_words)


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
