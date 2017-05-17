
import itertools


# similarity
def jaccard(ng1, ng2):
    if ng1.keys().isdisjoint(ng2.keys()):
        return 0
    intersection = set(ng1.keys()).intersection(set(ng2.keys()))
    return len(intersection) / (len(ng1) + len(ng2))


def weighted_jaccard(ng1, ng2):
    ng1_set = set(ng1.keys())
    ng2_set = set(ng2.keys())
    intersection = ng1_set.intersection(ng2_set)

    return sum(min(ng1[i], ng2[i]) for i in intersection) /\
        (sum(max(ng1[i], ng2[i]) for i in intersection) +
         sum(ng1[i] for i in ng1_set.difference(ng2_set)) +
         sum(ng2[i] for i in ng2_set.difference(ng1_set)))


def filter_ngrams(ngrams_dict, n=None):
    if n is None:
        return ngrams_dict
    return {k: v for k, v in ngrams_dict.items() if len(k) in n}


# computations
def intra_jaccard(authors_ng, n=None):
    total_jacc, pairs = 0, 0
    authors_ng_list = list(authors_ng.items())
    for a1_idx in range(len(authors_ng_list)):
        for a2_idx in range(a1_idx + 1):
            if authors_ng_list[a1_idx][0] == authors_ng_list[a2_idx][0]:
                continue
            pairs += 1
            total_jacc += jaccard(
                filter_ngrams(authors_ng_list[a1_idx][1].indices, n=n),
                filter_ngrams(authors_ng_list[a2_idx][1].indices, n=n))
    return total_jacc / pairs


def all_combinations(n_list):
    out = []
    for n in range(1, len(n_list) + 1):
        out += list(itertools.combinations(n_list, n))
    return out


def per_author_jaccard(gener_ng, discrim_ng, n=None):
    out = {}
    for author in gener_ng:
        out[author] = jaccard(
            filter_ngrams(gener_ng[author].indices, n=n),
            filter_ngrams(discrim_ng[author].indices, n=n))
    return out


# ngram extraction
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

    def normalize(self):
        total = sum(self.indices.values())
        self.indices = {k: v / total for k, v in self.indices.items()}
        return self
