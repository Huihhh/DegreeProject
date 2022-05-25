from typing import Union
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import wandb
import torch
import numpy as np
from utils.utils import hammingDistance

class HDistanceLogger(Callback):

    def __init__(self, log_every: Union[int, list]) -> None:
        '''
        Log the Hamming distance between untrained model and trained models at specific stages

        Parameter
        ----------
        * log_every: int or list of int
        '''
        super().__init__()
        self.log_every = log_every
        assert isinstance(log_every, int) or isinstance(log_every, list) or isinstance(self.log_every, np.ndarray), 'invalid type of log_every, must be list or int'

    def on_train_epoch_end(self, trainer: 'pl.trainer', pl_module: 'pl.LightningModule') -> None:
        current_epoch = trainer.current_epoch
        if isinstance(self.log_every, int):
            iflog = (current_epoch + 1) % self.log_every == 0
        else:
            iflog = current_epoch in self.log_every

        if iflog:
            sigs_grid = torch.unique(pl_module.grid_sigs, dim=0)
            hdistance = hammingDistance([sigs_grid, pl_module.grid_sigs0], pl_module.device).diag()
            avg_Hdistance = hdistance.mean() # ? mean?
            wandb.log({
                'epoch': pl_module.current_epoch, 
                'avg. Hamming distance': avg_Hdistance, 
                'distribution': wandb.Histogram(hdistance.cpu())
            })

