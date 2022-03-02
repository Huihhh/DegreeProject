from pytorch_lightning.callbacks import Callback
import wandb
import torch
import os
import hydra
import numpy as np
from utils.get_signatures import get_signatures
from utils.utils import hammingDistance
from models import *

class HDistanceLogger(Callback):

    def __init__(self, log_every, dataset, CFG, input_dim) -> None:
        super().__init__()
        self.log_every = log_every
        assert isinstance(log_every, int) or isinstance(log_every, list) or isinstance(self.log_every, np.ndarray), 'invalid type of log_every, must be list or int'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grid_points, _ = dataset.grid_data
        self.grid_points = torch.tensor(grid_points).float().to(device)

        untrained_model = MODEL[CFG.MODEL.name](input_dim=input_dim,seed=CFG.seed, **CFG.MODEL)
        untrained_model = untrained_model.to(device)
        _, sigs_grid_0, _ = get_signatures(self.grid_points, untrained_model)
        self.sigs_grid_0 = torch.unique(sigs_grid_0, dim=0)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        current_epoch = trainer.current_epoch
        if isinstance(self.log_every, int):
            iflog = current_epoch % self.log_every == 9
        else:
            iflog = current_epoch in self.log_every

        if iflog:
            # description = f'trained model after epoch {current_epoch}'
            # model_artifact = wandb.Artifact(f'model_epoch{current_epoch}', type='model', description=description)
            # torch.save(pl_module.model, self.LOCAL_MODEL_FILE)
            # model_artifact.add_file(self.LOCAL_MODEL_FILE)
            # wandb.run.log_artifact(model_artifact)


            _, sigs_grid, _ = get_signatures(self.grid_points, pl_module.model)
            sigs_grid = torch.unique(sigs_grid, dim=0)
            hdistance = hammingDistance([sigs_grid, self.sigs_grid_0], pl_module.device).diag()
            avg_Hdistance = hdistance.mean() # ? mean?
            wandb.log({'epoch': pl_module.current_epoch, 'avg. Hamming distance': avg_Hdistance, 'distribution': wandb.Histogram(hdistance.cpu())})

