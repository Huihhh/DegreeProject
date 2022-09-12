from typing import Any, Callable
import os
import sys
import logging
import pytorch_lightning as pl
import torch
from torch import optim
import numpy as np
import wandb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.ema import EMA
from utils.utils import accuracy
from utils.lr_schedulers import get_cosine_schedule_with_warmup
from utils.get_signatures import get_signatures

logger = logging.getLogger(__name__)

class Bicalssifier(pl.LightningModule):
    def __init__(self, model: Any, dataset: 'pl.LightningDataModule',
        n_epoch: int=100,
        th: float=0.5, 
        lr: float=0.01,
        use_scheduler: bool=True,
        warmup: int=0,
        wdecay: float=0.01,
        batch_size: int=32,
        ema_used: bool=False, 
        ema_decay: float=0.9,
        **kwargs
    ) -> None:
        '''
        Training process for biclassification using sigmoid.

        Parameter
        ----------
        * model: NN model
        * dataset: pytorch-lightning LightningDataModule
        * n_epoch: number of training epochs
        * th: threshold on sigmoid for prediction
        * lr: learning rate
        * warmup: warmup epochs
        * wdecay: weight decay
        * batch_size: default 32
        * ema_used: if use ema
        * ema_decay: ema decay rate

        '''
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.grid_points, self.grid_labels = dataset.grid_data
        self.N_EPOCH = n_epoch
        self.LR = lr
        self.USE_SCHEDULER = use_scheduler
        self.WDECAY = wdecay
        self.BATCH_SIZE=batch_size
        self.WARMUP = warmup
        self.TH = th
        
        self.init_criterion()
        # used EWA or not
        self.EMA = ema_used
        if self.EMA:
            self.EMA_DECAY = ema_decay,
            self.ema_model = EMA(self.model, ema_decay)
            logger.info("[EMA] initial ")
        

    def init_criterion(self) -> Callable:
        '''
        Generate loss function
        '''
        def compute_loss(y_pred, y):
            total_loss = torch.nn.BCELoss()(y_pred, y)
            return {'total_loss': total_loss}
        self.criterion = compute_loss

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.WDECAY},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # optimizer = optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.WDECAY)
        optimizer = optim.Adam(grouped_parameters, lr=self.LR)
        #    momentum=self.CFG.optim_momentum, nesterov=self.CFG.used_nesterov)
        if self.USE_SCHEDULER:
            steps_per_epoch = np.ceil(len(self.dataset.trainset) / self.BATCH_SIZE)  # eval(self.CFG.steps_per_epoch)
            total_training_steps = self.N_EPOCH * steps_per_epoch
            warmup_steps = self.WARMUP * steps_per_epoch
            scheduler = {
                'scheduler': get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps),
                'interval': 'step',
                # 'strict': True,
            }
            return [optimizer], [scheduler]
        return optimizer

    def forward(self, x):
        return self.model.forward(x)

    def on_post_move_to_device(self):
        super().on_post_move_to_device()
        # used EWA or not
        # init ema model after model moved to device
        if self.EMA:
            self.ema_model = EMA(self.model, self.EMA_DECAY)
            logger.info("[EMA] initial ")

    def on_fit_start(self) -> None:
        self.grid_points = self.grid_points.to(self.device)


    def on_train_epoch_start(self) -> None:
        self.net_out, self.grid_sigs, _ = get_signatures(self.grid_points, self.model, self.device)
        if self.current_epoch == 0:
            self.grid_sigs0 = torch.unique(self.grid_sigs, dim=0)
        self.log('#Linear regions', len(torch.unique(self.grid_sigs, dim=0)), on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_pred, _ = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        y_pred = torch.where(y_pred > self.TH, 1.0, 0.0)
        acc = accuracy(y_pred, y)
        for name, metric in losses.items():
            self.log(f'train.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('train.acc', acc, on_step=False, on_epoch=True)
        return losses['total_loss']

    def training_step_end(self, loss, *args, **kwargs):
        if self.EMA:
            self.ema_model.update_params()
        return loss

    def on_train_epoch_end(self) -> None:
        if self.EMA:
            self.ema_model.update_buffer()
            self.ema_model.apply_shadow()

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_pred, _ = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        y_pred = torch.where(y_pred > self.TH, 1.0, 0.0)
        acc = accuracy(y_pred, y)
        for name, metric in losses.items():
            self.log(f'val.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('val.acc', acc, on_step=False, on_epoch=True)
        return acc

    def validation_epoch_end(self, *args, **kwargs):
        if self.EMA:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_pred, _ = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        y_pred = torch.where(y_pred > self.TH, 1.0, 0.0)
        acc = accuracy(y_pred, y)
        self.log('test', {**losses, 'acc': acc})
        return acc

