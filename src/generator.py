#!/usr/bin/env

import os
import random
import copy

import torch.nn as nn

from seqmod.modules import LM
from seqmod.misc.trainer import LMTrainer
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.loggers import StdLogger
from seqmod.misc.dataset import BlockDataset, Dict
from seqmod.utils import save_model


def make_generator_hook(max_words=100):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.model.eval()
        trainer.log("info", "Generating %d-words long doc..." % max_words)
        doc, score = trainer.model.generate_doc(max_words=max_words)
        trainer.log("info", '\n***\n' + '\n'.join(doc) + "\n***")
        trainer.model.train()

    return hook


class LMGenerator(LM):
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
            optim_method='SGD', lr=1., max_norm=5.,
            start_decay_at=15, decay_every=5, lr_decay=0.8,
            # other parameters
            gpu=False, verbose=True, add_hook=False):
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
        self.gpu = gpu
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

    @staticmethod
    def fit_vocab(examples, max_size=None, min_freq=1):
        d = Dict(eos_token='<eos>', bos_token='<bos>', force_unk=False,
                 max_size=max_size, min_freq=min_freq)
        d.fit(examples)
        return d

    def copy(self):
        self.cpu()     # ensure model is in cpu to avoid exploding gpu
        return copy.deepcopy(self)

    def generate_doc(self, max_words=2000, max_sent_len=200, reset_every=10,
                     max_tries=5, **kwargs):
        """
        Parameters
        ===========
        max_words : int, max number of words in the output document
        max_sent_len : int, max number of characters per generated sentence
        reset_every : int, number of sentences to wait until resetting hidden
            state with the encoding of a randomly selected training sentence
        max_tries : int, number of attempts at generating a well-formed sentence
            before returning (well-formed meaning that it ends with <eos>).
        kwargs : rest arguments passed onto LM.generate
        """

        def generate_sent(max_tries=5, **kwargs):
            tries, hyp = 0, []
            while (not hyp or hyp[-1] != self.d.eos_token) and tries < max_tries:
                tries += 1
                scores, hyps = self.generate(
                    self.d, max_seq_len=max_sent_len, gpu=self.gpu, **kwargs)
                score, hyp = scores[0], hyps[0]
            sent = ''.join(self.d.vocab[c] for c in hyp)
            sent = sent.replace('<bos>', '').replace('<eos>', '')
            return sent, score

        sent, score = generate_sent(max_tries=max_tries, **kwargs)
        seed_text, doc, words = None, [sent], 0
        while words < max_words:
            kwargs.update({'seed_text': seed_text})  # use user seed only once
            sent, sent_score = generate_sent(max_tries=max_tries, **kwargs)
            doc.append(sent)
            words += len(sent.split())
            score += sent_score
            if len(doc) % reset_every == 0:
                # reset seed to randomly picked training sentence
                sent_idx = random.randint(0, len(self.examples) - 1)
                seed_text = self.examples[sent_idx]

        return doc, score


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--subset', default='PL')
    parser.add_argument('--crop_docs', default=False, type=int,
                        help='Maximum nb of words per document')
    parser.add_argument('--foreground_authors',
                        default=('Augustinus Hipponensis',
                                 'Hieronymus Stridonensis',
                                 'Bernardus Claraevallensis',
                                 'Walafridus Strabo'),
                        type=lambda args: args.split(','))
    parser.add_argument('--save_path', default='')
    parser.add_argument('--load_data', default='')
    # train
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--bptt', default=50, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--add_hook', action='store_true')
    # model
    parser.add_argument('--emb_dim', default=24, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    args = parser.parse_args()

    from data import DataReader, crop_docs
    if args.load_data:
        reader = DataReader.load(args.load_data)
    else:
        reader = DataReader(
            name=args.subset, foreground_authors=args.foreground_authors)
    train, _, _ = reader.foreground_splits()  # take gener split
    X_authors, _, X_train = train             # ignore titles
    if args.crop_docs:
        X_train = list(crop_docs(X_train, max_words=args.crop_docs))
    fitted_d = LMGenerator.fit_vocab([sent for doc in X_train for sent in doc])
    vocab = len(fitted_d)

    subpath = 'experiments/%s' % args.save_path
    if not os.path.isdir(subpath):
        os.mkdir(subpath)
    if args.save_path and not args.load_data:  # only save if not loaded
        # save reader (with splits)
        authors = ['-'.join(author.replace('.', '').split())
                   for author in args.foreground_authors]
        fname = '{name}.{foreground_authors}'.format(
            name=args.subset, foreground_authors='_'.join(authors))
        reader.save(os.path.join(subpath, fname))

    for author in set(X_authors):
        generator = LMGenerator(
            vocab, args.emb_dim, args.hid_dim, args.num_layers, dropout=0.3)
        examples = [sent for doc_author, doc in zip(X_authors, X_train)
                    for sent in doc if doc_author == author]
        n_words, n_sents = sum(len(s.split()) for s in examples), len(examples)
        print('Training %s on %d words, %d sents' % (author, n_words, n_sents))
        try:
            generator.fit(
                examples, fitted_d, args.batch_size, args.bptt, args.epochs,
                gpu=args.gpu, add_hook=args.add_hook)
            generator.eval()        # set to validation mode
            if args.save_path:
                model_path = '%s/%s' % (subpath, '_'.join(author.split()))
                save_model(generator, model_path)
        except Exception as e:
            print("Couldn't train %s. Exception: %s" % (author, str(e)))
