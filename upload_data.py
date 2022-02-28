from numpy import get_array_wrap
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
import wandb
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets.dataset import Dataset
from models import *
from experiments.base_trainer import Bicalssifier
from utils import flat_omegadict, set_random_seed


 
@hydra.main(config_path='./config', config_name='sampleEfficiency')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)

    # # For reproducibility, set random seed
    set_random_seed(CFG.seed)

    if CFG.DATASET.name == 'eurosat':
        dataset = Dataset(seed=CFG.seed, **CFG.DATASET)
    else:
        # get datasets
        dataset = Dataset(seed=CFG.seed, **CFG.DATASET)
    
    run = wandb.init(project=CFG.wandb_project, job_type="upload")

    for phase in ['train', 'val', 'test']:
        datasize = getattr(dataset, f'n_{phase}')
        DATA_AT = "_".join([CFG.DATASET.name, phase, str(datasize)]) # e.g., Spiral_train_194
        data_at = wandb.Artifact(DATA_AT, type=f"{phase}_data")
        # TODO: to be finished, but is it necessary to upload the data artifact?
    wandb.finish()




if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
