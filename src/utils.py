
import os


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
