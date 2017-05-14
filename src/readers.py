
import re
import os
import json
import functools
from collections import namedtuple, defaultdict

from lxml import etree

from cltk.tokenize.sentence import TokenizeSentence


def _fetch_latin_models():
    print("Fetching cltk tokenizers...")
    from cltk.corpus.utils.importer import CorpusImporter
    CorpusImporter('latin').import_corpus('latin_models_cltk')


ROOT_FOLDER = './data/'
DOC = namedtuple('DOC', ['author', 'title', 'sentences', 'nb_words'])
try:
    CLTK_TOK = TokenizeSentence('latin')
except:
    _fetch_latin_models()


def detokenizer(tokens):
    post_punc, pre_punc = {';', '.', ',', ':', '?', '!', ')'}, {'('}
    def func(acc, x):
        if x in post_punc:
            return acc + x
        if acc[-1] in pre_punc:
                return acc + x
        else:
            return acc + ' ' + x
    return functools.reduce(func, tokens)


def packhum_sentence_tokenizer(doc):
    # remove verse line-markers
    doc = re.sub(r"[0-9]+(\.)?([0-9]+)?", "", doc)
    # normalize whitespace
    doc = re.sub(r"(\n[ ]+)+", "\n", doc)
    return CLTK_TOK.tokenize_sentences(doc)


def _reader_gen(doc_func, root, include, exclude, subpath, min_sent_len):
    found = set()
    for f in os.listdir(os.path.join(root, subpath)):
        author = f.split('.')[0].replace('_', ' ')
        if (exclude and author not in exclude) or \
           (include and author in include) or \
           (not include and not exclude):
            with open(os.path.join(root, subpath, f), 'r+') as inf:
                title, nb_words, sentences = doc_func(inf, min_sent_len)
            yield DOC(author=author, title=title,
                      nb_words=nb_words, sentences=sentences)
            found.add(author)
    # check whether all include authors have been found
    for author in include:
        assert author in found, "Didn't found author %s" % author


def _packhum_func(inf, min_sent_len):
    work = json.load(inf)
    title = work['author']
    sentences = [s for page in work['pages']
                 for s in packhum_sentence_tokenizer(page['text'])
                 if len(s.split()) > min_sent_len]
    nb_words = sum(len(s.split()) for s in sentences)
    return title, nb_words, sentences


def packhum_reader(root=ROOT_FOLDER, include=(), exclude=(),
                   subpath='packhum/merged', min_sent_len=5):
    """
    Parameters
    ===========

    root : str, top folder of the processed data
    exclude : tuple of str, authors to skip when reading. Note that
        author names are determined by the file name substituting 
        underscores `_` with blankspaces ` `.
    indlude : tuple of str, authors to include when reading.
    """
    return _reader_gen(
        _packhum_func, root, include, exclude, subpath, min_sent_len)


def pl_sentence_tokenizer(doc, min_sent_len=5):
    """
    Transform .vrt files into list of sentences.
    Original sentences are markup <s></s>. However, it seems that
    in some document sentences have been automatically terminated with a ".",
    and the actual final punctuation has been segmented to its own <s> element.
    Example:
        <s>
        Quem	PRO	qui2
        ...
        Domine	SUB	domina
        .	SENT	.
        </s>
        <s>
        .	SENT	.
        </s>

    as well as:

        <s>
        Nam	CON	nam
        ...
        est	VBE	sum
        pedissequa	QLF	pedisequus2
        :	PON	:
        .	SENT	.
        </s>
        <s>
        Te	PRO	tu
        ...
    Therefore we remove all last sentence tokens and add all single-token
    sentences to the previous <s> element.
    """
    sents, out, eos_only = [], [], 0
    for s in doc:
        lines = s.text.strip().split('\n')
        sent = [line.split('\t')[0] for line in lines]
        if len(sent) == 1:
            eos_only += 1
        sents.append(sent)

    if eos_only > (len(sents) / 3):  # take a 1/3 ratio as threshold
        for idx, s in enumerate(sents[:-1]):
            if len(sents[idx]) > 1 and len(sents[idx]) > min_sent_len:
                if len(sents[idx + 1]) == 1:
                    # assume doubled eos only if followed by eos only sent
                    out.append(sents[idx][:-1] + sents[idx + 1])
                else:
                    out.append(sents[idx])
        if len(sents[-1]) > 1 and len(sents[-1]) > min_sent_len:
            out.append(sents[-1])
    else:
        out = sents

    return [detokenizer(s) for s in out]


def _pl_func(inf, min_sent_len):
    s = inf.read()
    tree_root = etree.fromstring(
        # get rid of rogue xml
        s.replace('<unknown>', 'unknown').encode('utf-8'))
    title = tree_root.attrib['titre']
    nb_words = tree_root.attrib['nb_tokens']
    sentences = pl_sentence_tokenizer(tree_root, min_sent_len=min_sent_len)
    nb_words = sum(len(s.split()) for s in sentences)
    return title, nb_words, sentences


def patrologia_reader(root=ROOT_FOLDER, include=(), exclude=(),
                      subpath='pl', min_sent_len=5):
    """
    Parameters
    ===========

    root : str, top folder of the processed data
    exclude : tuple of str, authors to skip when reading. Note that
        author names are determined by the file name substituting 
        underscores `_` with blankspaces ` `.
    indlude : tuple of str, authors to include when reading.
    """
    return _reader_gen(
        _pl_func, root, include, exclude, subpath, min_sent_len)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--min_sent_len', default=5, type=int)
    args = parser.parse_args()

    def word_counts(docs):
        out = defaultdict(lambda: defaultdict(int))
        for doc in docs:
            out[doc.author][doc.title] += int(doc.nb_words)
        return out

    def print_wc(wc, pad=10):
        max_author = max(len(a) for a in wc)
        print('{author},{mean_words},{words},{docs}'.format(
            author='author'.replace('.', ' ').ljust(max_author, ' '),
            mean_words='mean_words'.ljust(pad, ' '),
            words='words'.ljust(pad, ' '),
            docs='docs'.ljust(pad, ' ')))
        for author in wc:
            words = sum(wc[author][d] for d in wc[author])
            docs = len(wc[author])
            print('{author},{mean_words},{words},{docs}'.format(
                author=author.replace('.', ' ').ljust(max_author, ' '),
                mean_words=str(words/docs).ljust(pad, ' '),
                words=str(words).ljust(pad, ' '),
                docs=str(docs).ljust(pad, ' ')))

    print_wc(word_counts(patrologia_reader(min_sent_len=args.min_sent_len)))
    print_wc(word_counts(packhum_reader(min_sent_len=args.min_sent_len)))
