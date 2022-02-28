from typing import Counter
import wandb
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch import linalg as LA
import torch.nn.functional as F

import math
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

from utils.get_signatures import get_signatures
from utils.ema import EMA
from utils.utils import accuracy
from utils.lr_schedulers import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

class Bicalssifier(pl.LightningModule):
    def __init__(self, model, dataset, CFG) -> None:
        super().__init__()
        self.model = model
        self.model_name = CFG.MODEL.name
        self.dataset = dataset
        self.CFG = CFG.EXPERIMENT
        self.init_criterion()

        # used EWA or not
        self.ema = self.CFG.ema_used
        if self.ema:
            self.ema_model = EMA(self.model, self.CFG.ema_decay)
            logger.info("[EMA] initial ")

    def init_criterion(self):
        def compute_loss(y_pred, y):
            total_loss = torch.nn.BCELoss()(y_pred, y)
            return {'total_loss': total_loss}
        self.criterion = compute_loss

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.CFG.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(grouped_parameters, lr=self.CFG.optim_lr)
        #    momentum=self.CFG.optim_momentum, nesterov=self.CFG.used_nesterov)
        steps_per_epoch = np.ceil(len(self.dataset.trainset) / self.CFG.batch_size)  # eval(self.CFG.steps_per_epoch)
        total_training_steps = self.CFG.n_epoch * steps_per_epoch
        warmup_steps = self.CFG.warmup * steps_per_epoch
        scheduler = {
            'scheduler': get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps),
            'interval': 'step',
            # 'strict': True,
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model.forward(x)

    def on_post_move_to_device(self):
        super().on_post_move_to_device()
        # used EWA or not
        # init ema model after model moved to device
        if self.CFG.ema_used:
            self.ema_model = EMA(self.model, self.CFG.ema_decay)
            logger.info("[EMA] initial ")


    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_pred, _ = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        y_pred = torch.where(y_pred > self.CFG.TH, 1.0, 0.0)
        acc = accuracy(y_pred, y)
        for name, metric in losses.items():
            self.log(f'tain.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('train.acc', acc, on_step=False, on_epoch=True)
        return losses['total_loss']
    


    def training_step_end(self, loss, *args, **kwargs):
        if self.CFG.ema_used:
            self.ema_model.update_params()
        return loss

    def on_train_epoch_end(self) -> None:
        # TODO: wandb watch model
        # for name, param in self.model.named_parameters():
        #     self.log(f'parameters/norm_{name}', LA.norm(param))
        # if self.current_epoch == self.CFG.n_epoch - 1: # log last epoch
                    # init grid points to plot linear regions
        grid_points, _ = self.dataset.grid_data
        _, sigs_grid, _ = get_signatures(torch.tensor(grid_points).float().to(self.device), self.model)
        num_lr = len(set([''.join(str(x) for x in s.tolist()) for s in sigs_grid])) 
        self.log(f'#linear regions', num_lr)
        del grid_points
        del sigs_grid
        if self.CFG.ema_used:
            self.ema_model.update_buffer()
            self.ema_model.apply_shadow()

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_pred, _ = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        y_pred = torch.where(y_pred > self.CFG.TH, 1.0, 0.0)
        acc = accuracy(y_pred, y)
        for name, metric in losses.items():
            self.log(f'val.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('val.acc', acc, on_step=False, on_epoch=True)
        return acc

    def validation_epoch_end(self, *args, **kwargs):
        if self.CFG.ema_used:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_pred, _ = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        y_pred = torch.where(y_pred > self.CFG.TH, 1.0, 0.0)
        acc = accuracy(y_pred, y)
        self.log('test', {**losses, 'acc': acc})
        return acc


    def resume_model(self, train_loader, val_loader, test_loader):
        if self.CFG.resume:
            if os.path.isfile(self.CFG.resume_checkpoints):
                logger.info(f"=> loading checkpoint '${self.CFG.resume_checkpoints}'")
                trainer = Trainer(resume_from_checkpoint=self.CFG.resume_checkpoints)
                trainer.fit(self, train_loader, val_loader)
                result = trainer.test(test_dataloaders=test_loader)
                logger.info("test", result)

