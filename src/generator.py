#!/usr/bin/env

import os
import math
import random
import copy
import multiprocessing
import functools
import operator
from collections import defaultdict, Counter

from joblib import Parallel, delayed

import torch.nn as nn

import numpy as np

from seqmod.modules import LM
from seqmod.misc.trainer import LMTrainer
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.loggers import StdLogger
from seqmod.misc.dataset import BlockDataset, Dict
from seqmod.utils import save_model, load_model

from src.data import DataReader
from src.utils import generate_docs, train_generator, crop_docs
from src.utils import sample, load_docs_from_dir


TINY = 1e-10


def make_generator_hook(max_words=100):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.model.eval()
        trainer.log("info", "Generating %d-words long doc..." % max_words)
        doc, score = trainer.model.generate_doc(max_words=max_words)
        trainer.log("info", '\n***\n' + '\n'.join(doc) + "\n***")
        trainer.model.train()

    return hook


class BasisGenerator(object):
    def fit(self, examples, d, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def fit_vocab(examples, max_size=None, min_freq=1):
        d = Dict(eos_token='<eos>', bos_token='<bos>', force_unk=False,
                 max_size=max_size, min_freq=min_freq)
        d.fit(examples)
        return d

    def copy(self):
        return copy.deepcopy(self)

    def save(self, path):
        save_model(self, path, mode='pickle')

    def generate_doc(self, max_words=2000, max_sent_len=200, reset_every=10,
                     max_tries=5, **kwargs):
        """
        Parameters
        ===========
        max_words : int, max number of words in the output document
        max_sent_len : int, max number of characters per generated sentence
        reset_every : int, number of sentences to wait until resetting hidden
            state with the encoding of a randomly selected training sentence
        max_tries : int, number of attempts at generating a well-formed sent
            before returning (well-formed meaning that it ends with <eos>).
        kwargs : rest arguments passed onto LM.generate
        """

        def generate_sent(max_tries=5, **kwargs):
            tries, hyp = 0, []
            while (not hyp or hyp[-1] != self.d.get_eos()) and tries < max_tries:
                tries += 1
                scores, hyps = self.generate(
                    self.d, max_seq_len=max_sent_len, **kwargs)
                score, hyp = scores[0], hyps[0]
            sent = ''.join(self.d.vocab[c] for c in hyp)
            sent = sent.replace('<bos>', '').replace('<eos>', '')
            return sent, score
        sent, score = generate_sent(max_tries=max_tries, **kwargs)
        seed_text, doc, words, scores = None, [sent], 0, [score]
        while words < max_words:
            kwargs.update({'seed_text': seed_text})  # use user seed only once
            sent, sent_score = generate_sent(max_tries=max_tries, **kwargs)
            doc.append(sent)
            words += len(sent.split())
            scores.append(sent_score)
            if len(doc) % reset_every == 0:
                # reset seed to randomly picked training sentence
                sent_idx = random.randint(0, len(self.examples) - 1)
                seed_text = self.examples[sent_idx]

        return doc, sum(scores) / len(scores)


class LMGenerator(LM, BasisGenerator):
    """
    Wrapper training and generating class for LM

    Parameters:
    ===========
    - vocab: int, vocabulary size.
    - emb_dim: int, embedding size,
        This value has to be equal to hid_dim if tie_weights is True and
        project_on_tied_weights is False, otherwise input and output
        embedding dimensions wouldn't match and weights cannot be tied.
    - hid_dim: int, hidden dimension of the RNN.
    - num_layers: int, number of layers of the RNN.
    - cell: str, one of GRU, LSTM, RNN.
    - bias: bool, whether to include bias in the RNN.
    - dropout: float, amount of dropout to apply in between layers.
    - tie_weights: bool, whether to tie input and output embedding layers.
        In case of unequal emb_dim and hid_dim values a linear project layer
        will be inserted after the RNN to match back to the embedding dim
    - att_dim: int, whether to add an attention module of dimension `att_dim`
        over the prefix. No attention will be added if att_dim is None or 0
    - deepout_layers: int, whether to add deep output after hidden layer and
        before output projection layer. No deep output will be added if
        deepout_layers is 0 or None.
    - deepout_act: str, activation function for the deepout module in camelcase
    """
    def fit(self,
            # dataset parameters
            examples, d, batch_size, bptt, epochs, split=0.1,
            # optimizer parameters
            optim_method='Adam', lr=0.001, max_norm=5.,
            start_decay_at=15, decay_every=5, lr_decay=0.8,
            # other parameters
            gpu=False, verbose=True, add_hook=False, **kwargs):
        """
        Parameters
        ===========
        examples : iterable of sentences (lists of strings)
        d : fitted Dict object,
            Use fit_vocab to get one. It should be the same that was used to
            estimate the vocab param in the constructor. Note that fit_vocab
            is a static method and doesn't need object initialization
        batch_size : int
        bptt : int, Backprop through time (max unrolling value)
        epochs : int
        """
        self.d = d
        self.examples = examples
        train, valid = BlockDataset(
            examples, self.d, batch_size, bptt, gpu=gpu).splits(
                test=split, dev=None)
        criterion = nn.CrossEntropyLoss()
        optim = Optimizer(
            self.parameters(), optim_method, lr=lr,
            lr_decay=lr_decay, start_decay_at=start_decay_at,
            decay_every=decay_every)
        trainer = LMTrainer(
            self, {'train': train, 'valid': valid}, criterion, optim)
        if verbose:
            trainer.add_loggers(StdLogger())
        checkpoints_per_epoch = max(len(train) // 10, 1)
        if add_hook:
            hooks_per_epoch = max(len(train) // (checkpoints_per_epoch * 1), 1)
            trainer.add_hook(make_generator_hook(), hooks_per_epoch)
        if gpu:
            self.cuda()
        trainer.train(epochs, checkpoint=checkpoints_per_epoch, gpu=gpu)

    def perplexity(self, examples):
        chars = [c for l in self.d.transform(examples) for c in l]
        return math.exp(-self.predict_proba(chars))

    def copy(self):
        self.cpu()     # ensure model is in cpu to avoid exploding gpu
        return super(self, LMGenerator).copy()

    def save(self, path):
        self.cpu()
        save_model(self, path)


class UnsmoothedLMGenerator(BasisGenerator):
    """
    Adapted from https://gist.github.com/yoavg/d76121dfde2618422139
    """
    def __init__(self, order=6):
        self.order = order
        self.lm = defaultdict(Counter)
        self.pad_token = '<bos>'

    def _ngrams(self, examples):
        data = [c for l in self.d.transform(examples) for c in l]
        data = [self.d.index(self.pad_token)] * self.order + data
        for i in range(len(data) - self.order):
            history, char = data[i:i+self.order], data[i+self.order]
            yield history, char

    def fit(self, examples, d, *args, **kwargs):
        self.examples = examples
        self.d = d

        for history, char in self._ngrams(examples):
            self.lm[tuple(history)][char] += 1

        def normalize(counter):
            s = float(sum(counter.values()))
            return {c: cnt/s for c, cnt in counter.items()}

        self.lm = {hist: normalize(chars) for hist, chars in self.lm.items()}

    def generate_char(self, seed, d):
        seed = seed[-self.order:]  # seed is ints
        dist = self.lm[tuple(seed)]
        values, probs = zip(*dist.items())
        idx = sample(np.array(list(probs)))
        score, char = probs[idx], values[idx]
        return score, char

    def generate(self, d, seed_text=None, max_seq_len=200, **kwargs):
        hyp, scores = [], []
        if seed_text is None or len(seed_text) < self.order:
            # seed is chars
            seed_text = [d.index(self.pad_token)] * self.order
        else:
            seed_text = [d.index(c) for c in seed_text][:self.order]
            hyp = seed_text
        for i in range(max_seq_len):
            score, char = self.generate_char(seed_text, d)
            seed_text = seed_text[-self.order:] + [char]
            hyp.append(char), scores.append(score)
            if char == d.get_eos():
                break
        return [functools.reduce(operator.mul, scores)], [hyp]

    def perplexity(self, examples):
        prob, nb_preds = 0., 0
        for history, char in self._ngrams(examples):
            nb_preds += 1
            try:
                dist = self.lm[tuple(history)]
                if not isinstance(dist, dict):  # legacy code
                    self.lm = {hist: {c: p for c, p in dists}
                               for hist, dists in self.lm.items()}
                    dist = dict(dist)
                char_prob = dist[char]
            except KeyError:
                # char_prob = TINY  # smooth
                nb_preds -= 1
                continue
            prob += math.log(char_prob)
        return math.exp(-prob / nb_preds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_path',
                        help='Required if generator_path is not given.')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--nb_words', default=2000, type=int,
                        help='Number of words per generated doc')
    parser.add_argument('--max_words', default=0, type=int,
                        help='Crop train docs to this number of words')
    parser.add_argument('--generate', action='store_true',
                        help='Whether to generate and store docs. ' +
                        'Text will be stored to save_path/generated')
    parser.add_argument('--nb_docs', type=int, default=50,
                        help='Number of generated docs per author')
    parser.add_argument('--generator_path', help='Path with ' +
                        'generators for loading (no training involved)')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--author_selection')
    # train
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--bptt', default=50, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optim_method', default='Adam')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--add_hook', action='store_true')
    # model
    parser.add_argument('--model', default='rnn_lm')
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--emb_dim', default=24, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.isdir(os.path.join(args.save_path, 'generators')):
        os.mkdir(os.path.join(args.save_path, 'generators'))

    keep_author = lambda author: True
    if args.author_selection:
        keep = set([author.replace('_', ' ')
                    for author in args.author_selection.split(',')])
        keep_author = lambda author: author in keep

    model_authors = {}
    if args.generator_path is not None:
        # load generators
        for f in os.listdir(args.generator_path):
            if not f.endswith('pt') and not f.endswith('pkl'):
                continue
            author = os.path.basename(f).split('.')[0].replace('_', ' ')
            if keep_author(author):
                model_authors[author] = os.path.join(args.generator_path, f)
    else:
        # load data
        if os.path.isdir(args.data_path):
            X_train, X_authors = load_docs_from_dir(args.data_path)
        else:                   # assume it's reader
            reader = DataReader.load(args.data_path)
            train, _, _ = reader.foreground_splits()  # take gener split
            X_authors, _, X_train = train             # ignore titles
        if args.max_words > 0:
            X_train = list(crop_docs(X_train, max_words=args.max_words))
        fitted_d = BasisGenerator.fit_vocab(
            [sent for doc in X_train for sent in doc])
        # train generators
        for author in set(X_authors):
            if not keep_author(author):
                continue
            examples = [sent for doc_author, doc in zip(X_authors, X_train)
                        for sent in doc if doc_author == author]
            if args.model == 'ngram_lm':
                generator = UnsmoothedLMGenerator(args.order)
            elif args.model == 'rnn_lm':
                generator = LMGenerator(
                    len(fitted_d), args.emb_dim, args.hid_dim, args.num_layers,
                    dropout=0.3)
            else:
                raise ValueError("Model must be ngram_lm or rnn_lm")
            n_w, n_s = sum(len(s.split()) for s in examples), len(examples)
            print('Training %s on %d words, %d sents' % (author, n_w, n_s))
            try:
                model_path = train_generator(
                    generator, author, examples, fitted_d, args)
                model_authors[author] = model_path
            except Exception as e:
                print("Couldn't train %s. Exception: %s" % (author, str(e)))

    # generation
    if args.generate:
        generated_path = '%s/generated/' % args.save_path
        if not os.path.isdir(generated_path):
            os.mkdir(generated_path)
        reset_every = 1 if args.model == 'ngram_lm' else 10
        Parallel(n_jobs=args.n_jobs)(
            delayed(generate_docs)(
                load_model(fpath), author, args.nb_docs, args.nb_words,
                save=True, path=generated_path, reset_every=reset_every)
            for author, fpath in model_authors.items())
