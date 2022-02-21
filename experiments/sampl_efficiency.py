from typing import Counter
import wandb
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import optim
from torch import linalg as LA
import torch.nn.functional as F

import numpy as np
import logging
from collections import defaultdict
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import *
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from .base_trainer import Bicalssifier
from utils.get_signatures import get_signatures

logger = logging.getLogger(__name__)


class SampleEffi(Bicalssifier):
    def __init__(self, model, dataset, CFG) -> None:
        super().__init__(model, dataset, CFG)
        # init grid points to plot linear regions
        self.grid_points, self.grid_labels = dataset.grid_data


    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.current_epoch == self.CFG.n_epoch - 1: # log last epoch
            _, sigs_grid, _ = get_signatures(torch.tensor(self.grid_points).float().to(self.device), self.model)
            num_lr = len(set([''.join(str(x) for x in s.tolist()) for s in sigs_grid])) 
            self.log(f'#linear regions', num_lr)






