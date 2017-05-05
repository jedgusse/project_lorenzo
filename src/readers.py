
import re
import os
import json
import string
import functools
# from xml.etree import ElementTree
from lxml import etree
from collections import namedtuple

from cltk.tokenize.sentence import TokenizeSentence


def _fetch_latin_models():
    print("Fetching cltk tokenizers...")
    from cltk.corpus.utils.importer import CorpusImporter
    CorpusImporter('latin').import_corpus('latin_models_cltk')


PARENT_FOLDER = './data/'
DOC = namedtuple('DOC', ['author', 'title', 'sentences', 'nb_words'])
try:
    CLTK_TOK = TokenizeSentence('latin')
except:
    _fetch_latin_models()


def detokenizer(tokens):
    def func(acc, x):
        if x not in string.punctuation:
            return acc + ' ' + x
        else:
            return acc + x
    return functools.reduce(func, tokens)


def packhum_sentence_tokenizer(doc):
    # remove verse line-markers
    doc = re.sub(r"[0-9]+(\.)?([0-9]+)?", "", doc)
    # normalize whitespace
    doc = re.sub(r"(\n[ ]+)+", "\n", doc)
    return CLTK_TOK.tokenize_sentences(doc)


def packhum_reader(parent=PARENT_FOLDER, exclude=(), include=()):
    with open(os.path.join(parent, 'packhum/packhum.json'), 'r+') as inf:
        for line in inf:
            obj = json.loads(line.strip())
            author, title = obj['author'], obj['work']
            if (exclude and author not in exclude) or \
               (include and author in include) or \
               (not include and not exclude):
                sentences = packhum_sentence_tokenizer(obj['text'])
                yield DOC(author=author,
                          title=title,
                          nb_words=sum(len(s.split()) for s in sentences),
                          sentences=sentences)


def pl_sentence_tokenizer(doc):
    """
    Transform .vrt files into list of sentences.
    Original sentences are markup <s></s>. However, it seems that
    all sentences have been automatically terminated with a ".",
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
    out, last = [], None
    for s in doc:
        lines = s.text.strip().split('\n')
        if len(lines) == 1:
            # . SENT . -only lines
            if last is not None:
                last.append(".")
            continue
        if last is not None:
            out.append(detokenizer(last))
        last = list(line.split('\t')[0] for line in lines[:-1])
    out.append(detokenizer(last))
    return out


def patrologia_reader(parent=PARENT_FOLDER, exclude=(), include=(),
                      subpath='patrologia_rnr'):
    for f in os.listdir(os.path.join(parent, subpath)):
        with open(os.path.join(parent, subpath, f), 'r+') as inf:
            string = inf.read()
            root = etree.fromstring(
                # get rid of rogue xml
                string.replace('<unknown>', 'unknown').encode('utf-8'))
            author, title = root.attrib['auteur'], root.attrib['titre']
            nb_words = root.attrib['nb_tokens']
            if (exclude and author not in exclude) or \
               (include and author in include) or \
               (not include and not exclude):
                sentences = pl_sentence_tokenizer(root)
                yield DOC(author=author,
                          title=title,
                          nb_words=nb_words,
                          sentences=sentences)
