from typing import Union
import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import wandb


class WeightLogger(Callback):
    def __init__(self, log_every:Union[int, list[int]]) -> None:
        '''
        Log the distribution of weights & bias using wandb.Histogram

        Parameter
        ----------------
        * log_every: int or list of int
        '''
        self.log_every = log_every

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        epoch = trainer.current_epoch
        if isinstance(self.log_every, int):
            iflog = epoch % self.log_every == 9
        else:
            iflog = epoch in self.log_every

        if iflog:
            for name, param in pl_module.model.named_parameters():
                pl_module.logger.experiment.log({f'historgram/{name}': wandb.Histogram(param.detach().cpu().view(-1))})
                pl_module.logger.experiment.log({f'variance/{name}': torch.var(param, unbiased=False)})