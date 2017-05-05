
import copy

import torch.nn as nn

from seqmod.modules import LM
from seqmod.misc.trainer import LMTrainer
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.loggers import StdLogger


class LMGenerator(LM):
    def train(self,
              # dataset parameters
              data, batch_size, epochs, splits,
              # optimizer parameters
              optim='SGD', lr=1., max_grad_norm=5.,
              start_decay_at=15, decay_every=5, lr_decay=0.8,
              # other parameters
              gpu=False):
        criterion = nn.CrossEntropyLoss()
        optim = Optimizer(self.parameters(), )
        trainer = LMTrainer(optim, criterion).train()
        trainer.add_loggers(StdLogger())

    def copy(self):
        self.cpu()              # ensure model is in cpu to avoid exploding gpu
        return copy.deepcopy(self)

    def generate_doc(self, training_data, d):
        pass
