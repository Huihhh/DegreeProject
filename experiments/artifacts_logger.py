from pytorch_lightning.callbacks import Callback
import wandb
import torch
import os
import hydra
import numpy as np

class ArtifactLogger(Callback):
    LOCAL_MODEL_DIR = '/outputs/checkpoints'

    def __init__(self, log_every) -> None:
        super().__init__()
        self.log_every = log_every
        assert isinstance(log_every, int) or isinstance(log_every, list) or isinstance(self.log_every, np.ndarray), 'invalid type of log_every, must be list or int'
        self.LOCAL_MODEL_DIR = hydra.utils.get_original_cwd() + self.LOCAL_MODEL_DIR
        os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
        self.LOCAL_MODEL_FILE = self.LOCAL_MODEL_DIR + '/model.h5'

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        current_epoch = trainer.current_epoch
        if isinstance(self.log_every, int):
            iflog = current_epoch % self.log_every == 9
        else:
            iflog = current_epoch in self.log_every

        if iflog:
            description = f'trained model after epoch {current_epoch}'
            model_artifact = wandb.Artifact(f'model_epoch{current_epoch}', type='model', description=description)
            torch.save(pl_module.model, self.LOCAL_MODEL_FILE)
            model_artifact.add_file(self.LOCAL_MODEL_FILE)
            wandb.run.log_artifact(model_artifact)