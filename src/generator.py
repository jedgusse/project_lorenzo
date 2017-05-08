
import random
import copy

import torch.nn as nn

from seqmod.modules import LM
from seqmod.misc.trainer import LMTrainer
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.loggers import StdLogger
from seqmod.misc.dataset import BlockDataset, Dict


class LMGenerator(LM):
    """
    Wrapper training function

    Parameters
    ===========
    examples: iterable of sentences (lists of strings)
    d: fitted Dict object,
        use fit_vocab to get one. It should be the same that was used to
        estimate the vocab param in the constructor. Note that fit_vocab
        is a static method and doesn't need object initialization.
    """
    def train(self,
              # dataset parameters
              examples, d, batch_size, bptt, epochs, split=0.1,
              # dict parameters
              max_size=None, min_freq=1,
              # optimizer parameters
              optim_method='SGD', lr=1., max_norm=5.,
              start_decay_at=15, decay_every=5, lr_decay=0.8,
              # other parameters
              gpu=False):
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
            self, {'train': train, 'valid': valid}, self.d, criterion, optim)
        trainer.add_loggers(StdLogger())
        trainer.train(epochs)

    @staticmethod
    def fit_vocab(examples, max_size=None, min_freq=1):
        d = Dict(eos_token='<eos>', max_size=max_size, min_freq=min_freq)
        d.fit(examples)
        return d

    def copy(self):
        self.cpu()              # ensure model is in cpu to avoid exploding gpu
        return copy.deepcopy(self)

    def generate_doc(self, max_words=5000, reset_every=10, **kwargs):
        sents, words, score = [], 0, 0

        def generate_sent(sents, words, score, **kwargs):
            scores, hyps = self.model.generate(
                self.d, max_seq_len=100, **kwargs)
            sent = ''.join([self.d.vocab[c] for c in hyps[0]])
            sent = sent.replace(self.d.get_eos(), '\n')
            sents.append(sent)
            words += len(sent.split())
            score += scores[0]
            return sents, words, score

        sents, words, score = generate_sent(sents, words, score, **kwargs)
        seed_text = None
        while words < max_words:
            kwargs.update({'seed_text': seed_text})  # use user seed only once
            sents, words, score = generate_sent(sents, words, score, **kwargs)
            if len(sents) % reset_every == 0:
                # reset seed to randomly picked training sentence
                seed_text = random.randint(0, len(self.examples) - 1)

        return sents, score
