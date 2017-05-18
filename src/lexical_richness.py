
"""
Implementation of K and Z lexical richness measures following (Tweedie &
Baayen 1998). In the paper it is shown that many of the proposed functions
can be grouped in two different families (one capturing the repeat rate,
the other capturing the vocabulary richness) and that K and Z are most
independent of text length inside their respective family.
"""


import math
from collections import Counter, defaultdict

import numpy as np


def permute_corpus(corpus):
    """
    Randomly permutes a corpus in word-form [w, w, w, ...]
    """
    return np.random.permutation(np.array(corpus))


def rank(corpus):
    """
    Computes a dictionary of frequencies of frequencies of frequencies.
    Word frequency -> that word frequency's frequency.
    """
    by_rank = defaultdict(int)
    for w, c in Counter(corpus).items():
        by_rank[c] += 1
    return by_rank


def lm(y, x):
    """
    Computes slope and intercept of the linear regression for two given inputs
    where `y` is the response and `x` is the predictor variable.
    """
    return np.linalg.lstsq(np.array([x, np.ones(len(x))]).T, np.array(y))[0]


def Z(corpus):
    """
    Implementation of the Zipf's rank-frequency distribution.
    Z is equivalent to the slope of the rank-frequency curve in the log-scale.
    """
    spectrum = np.array(sorted(rank(corpus).items(), reverse=True))
    ranked = np.cumsum(spectrum[:, 1])
    return lm(
        np.log(spectrum[:, 0]),
        np.log(ranked)
    )[0]


def K(corpus):
    """
    Implementation of Yule's K $10^3 [-1/N + \sigma_i V(i, N) (i/N)^2]$
    where i iterates over frequencies of word frequencies in the input corpus,
    V(i, N) if the frequency of the ith frequency in the corpus, and N
    is the length of the corpus in tokens.
    """
    term = sum(n_types * math.pow(r / len(corpus), 2)
               for r, n_types in rank(corpus).items())
    return 10e3 * (-1/len(corpus) + term)


def trajectory(corpus, func, n_chunks):
    """
    Computes lexical trajectory over steps following partition by `n_chunks`
    """
    perm = permute_corpus(corpus)
    step = len(corpus) // n_chunks
    return [(i, func(perm[:i])) for i in list(range(0, len(corpus), step))[1:]]
