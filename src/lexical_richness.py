
import math
from collections import Counter, defaultdict

import numpy as np


def permuted_author(sents):
    corpus = np.array([w for s in sents for w in s])
    return np.random.permutation(corpus)


def rank(corpus):
    by_rank = defaultdict(int)
    for w, c in Counter(corpus).items():
        by_rank[c] += 1
    return by_rank


def K(corpus):
    term = sum(n_types * math.pow(r / len(corpus), 2)
               for r, n_types in rank(corpus).items())
    return 10e4 * (-1/len(corpus) + term)


def Z(corpus):
    pass
