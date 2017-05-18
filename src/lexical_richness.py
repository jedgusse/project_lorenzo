
"""
Implementation of K and Z lexical richness measures following (Tweedie &
Baayen 1998). In the paper it is shown that many of the proposed functions
can be grouped in two different families (one capturing the repeat rate,
the other capturing the vocabulary richness) and that K and Z are most
independent of text length inside their respective family.
"""


import math
from collections import Counter, defaultdict

import joblib
import multiprocessing
import numpy as np


def permute_corpus(corpus, rng=None):
    """
    Randomly permutes a corpus in word-form [w, w, w, ...]
    """
    if rng is not None:
        return rng.permutation(np.array(corpus))
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


def trajectory(corpus, func, n_chunks, rng=None):
    """
    Computes lexical trajectory over steps following partition by `n_chunks`
    """
    perm = permute_corpus(corpus, rng)
    step = len(corpus) // n_chunks
    return [(i, func(perm[:i])) for i in list(range(0, len(corpus), step))[1:]]


def mc_interval(samples, interval=0.025):
    """
    Monte Carlo confidence interval on an input vector of samples.
    Returns lower, mean and upper values for the interval.
    """
    samples = np.array(samples)
    split_idx = int(len(samples) * interval)
    sort = np.sort(samples)
    return sort[split_idx], sort.mean(), sort[-split_idx]


def bootstrap_trajectory(corpus, func, n_chunks, resampling=1000):
    """
    Computes a `resampling` number of samples for the given richness
    `func` at each of the steps resulting from chunking `corpus` in
    `n_chunks` chunks.

    Return
    ------
    [(step, (score_1, score_2, score_resampling)), ...]
    """
    seeds = [r * np.random.randint(10) for r in range(resampling)]
    # compute bootstrapped values
    result = [trajectory(corpus, func, n_chunks) for _ in range(resampling)]
    num_cores = multiprocessing.cpu_count()
    result = joblib.Parallel(n_jobs=num_cores)(
        joblib.delayed(trajectory)(
            corpus, func, n_chunks, rng=np.random.RandomState(seeds[r])
        ) for r in range(resampling))
    # compute confidence intervals
    return [(step[0][0], mc_interval([score for _, score in step]))
            for step in zip(*result)]
