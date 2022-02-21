import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
import wandb
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from experiments.artifacts_logger import ArtifactLogger




from datasets.dataset import Dataset
from models import *
from experiments.experiment_multiclass import ExperimentMulti
from experiments.litExperiment import LitExperiment
from experiments.sampl_efficiency import SampleEffi
from experiments.base_trainer import Bicalssifier
from utils import flat_omegadict
 
@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))

    # # For reproducibility, set random seed
    if CFG.Logging.seed == 'None':
        CFG.Logging.seed = random.randint(1, 10000)
    random.seed(CFG.Logging.seed)
    os.environ['PYTHONHASHSEED'] = str(CFG.Logging.seed)
    np.random.seed(CFG.Logging.seed)
    torch.manual_seed(CFG.Logging.seed)
    torch.cuda.manual_seed_all(CFG.Logging.seed)
    pl.seed_everything(CFG.Logging.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if CFG.DATASET.name == 'eurosat':
        model = MODEL[CFG.MODEL.name](**CFG.MODEL)
        dataset = Dataset(resnet=model.resnet, **CFG.DATASET)
        input_dim = model.fcs[0].fc.in_features
    else:
        # get datasets
        dataset = Dataset(**CFG.DATASET)
        input_dim = dataset.trainset[0][0].shape[0]

        # build model
        model = MODEL[CFG.MODEL.name](input_dim=input_dim, **CFG.MODEL)
    logger.info("[Model] Building model -- input dim: {}, hidden nodes: {}, out dim: {}"
                                .format(input_dim, CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    if CFG.EXPERIMENT.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)

    # if CFG.MODEL.name in ['resnet', 'sResnet']:
    #     experiment = ExperimentMulti(model, dataset, CFG)
    # else:
    #     experiment = LitExperiment(model, dataset, CFG)
    experiment = Bicalssifier(model, dataset, CFG)

    wandb_logger = WandbLogger(
        project=CFG.EXPERIMENT.wandb_project,
        name=CFG.EXPERIMENT.name,
        config=flat_omegadict(CFG),
    )
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val.total_loss',
        dirpath='checkpoints/',
        filename='degree-project-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # artifact_logger = ArtifactLogger()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if CFG.early_stop:
        callbacks.append(EarlyStopping('val.total_loss', min_delta=0.0001, patience=10, mode='min', strict=True))

    trainer = Trainer(
        # accelerator="ddp",  # if torch.cuda.is_available() else 'ddp_cpu',
        callbacks=callbacks,
        logger=wandb_logger,
        # checkpoint_callback=False if CFG.debug else checkpoint_callback,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=CFG.EXPERIMENT.n_epoch,
        # gradient_clip_val=1,
        progress_bar_refresh_rate=0
    )
    logger.info("======= Training =======")

    trainer.fit(experiment, dataset)
    logger.info("======= Testing =======")
    trainer.test(experiment, datamodule=dataset)
    wandb.finish()




if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
