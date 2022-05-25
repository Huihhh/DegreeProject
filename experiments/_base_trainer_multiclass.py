from typing import Any, Callable
import os
import sys
import logging
import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import wandb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.ema import EMA
from utils.utils import accuracy, acc_topk
from utils.lr_schedulers import get_cosine_schedule_with_warmup
from utils.get_signatures import get_signatures

logger = logging.getLogger(__name__)

class Multicalssifier(pl.LightningModule):
    def __init__(self, model: Any, dataset: 'pl.LightningDataModule',
        n_epoch: int=100,
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
        self.n_epoch = n_epoch
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.wdecay = wdecay
        self.batch_size=batch_size
        self.warmup = warmup

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
            total_loss = F.cross_entropy(y_pred, y)
            return {'total_loss': total_loss}
        self.criterion = compute_loss

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        
        if self.use_scheduler:
            steps_per_epoch = np.ceil(len(self.dataset.trainset) / self.batch_size)  # eval(self.CFG.steps_per_epoch)
            total_training_steps = self.n_epoch * steps_per_epoch
            warmup_steps = self.warmup * steps_per_epoch
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


    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model.forward(x)
        losses = self.criterion(out, y)
        acc, = acc_topk(out, y)
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
        x, y = batch
        out = self.model.forward(x)
        losses = self.criterion(out, y)
        acc, = acc_topk(out, y)
        for name, metric in losses.items():
            self.log(f'val.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('val.acc', acc, on_step=False, on_epoch=True)
        return acc

    def validation_epoch_end(self, *args, **kwargs):
        if self.EMA:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model.forward(x)
        losses = self.criterion(out, y)
        acc, = acc_topk(out, y)
        self.log('test', {**losses, 'acc': acc})
        return acc

