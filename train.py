import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from copy import deepcopy
import wandb
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from experiments.hDistance_logger import HDistanceLogger
from experiments.linearRegion_logger import LinearRegionLogger
from experiments.artifacts_logger import ArtifactLogger

from datasets.dataset import Dataset
from experiments.weight_logger import WeightLogger
from models import *
from experiments._base_trainer import Bicalssifier
from experiments._base_trainer_multiclass import Multicalssifier
from utils import flat_omegadict, set_random_seed


@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))
    if (device_n := CFG.CUDA_VISIBLE_DEVICES) is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_n)
        print('CUDA_VISIBLE_DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device('cuda' if torch.cuda.is_available() and CFG.use_gpu else 'cpu')

    # # For reproducibility, set random seed
    set_random_seed(CFG.seed)
    # get datasets
    dataset = Dataset(seed=0, **CFG.DATASET) #* fix dataset
    input_dim = dataset.trainset[0][0].shape[0]
    
    # build model
    model = MODEL[CFG.MODEL.name](input_dim=input_dim,seed=CFG.seed, **CFG.MODEL)
    logger.info("[Model] Building model -- input dim: {}, hidden nodes: {}, out dim: {}"
                                .format(input_dim, CFG.MODEL.h_nodes, CFG.MODEL.out_dim))
    model = model.to(device=device)

    if dataset.name.lower() == 'eurosat':
        LitModel = Multicalssifier
    else :
        LitModel = Bicalssifier    
    litModel = LitModel(model, dataset, batch_size=CFG.DATASET.batch_size, **CFG.TRAIN)
    
    wandb_logger = WandbLogger(
        project=CFG.wandb_project,
        name=CFG.run_name,
        config=flat_omegadict(CFG),
        job_type='train'
    )
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val.total_loss',
        dirpath='checkpoints/',
        filename='degree-project-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )


    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]

    if CFG.early_stop:
        callbacks.append(EarlyStopping('val.total_loss', min_delta=0.0001, patience=10, mode='min', strict=True))

    if (log_every := CFG.log_hdistance_every) is not None:
        # log_every = np.hstack([np.arange(1, 10), np.arange(19, 100, 10), np.arange(199, 1000, 100)])
        hdis_logger = HDistanceLogger(log_every=log_every)
        callbacks.append(hdis_logger)

    if (log_every := CFG.log_LRimg_every) is not None:
        lr_logger = LinearRegionLogger(log_every, dataset.grid_data, dataset.trainset.tensors)
        callbacks.append(lr_logger)

    if (log_every := CFG.log_artifact_every) is not None:
        artifact_logger = ArtifactLogger(log_every)
        callbacks.append(artifact_logger)
    
    if (log_every := CFG.log_weight_every) is not None:
        weight_logger = WeightLogger(log_every)
        callbacks.append(weight_logger)
        
    trainer = Trainer(
        # accelerator="ddp",  # if torch.cuda.is_available() else 'ddp_cpu',
        callbacks=callbacks,
        logger=wandb_logger,
        # checkpoint_callback=False if CFG.debug else checkpoint_callback,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=CFG.TRAIN.n_epoch,
        # gradient_clip_val=1,
        progress_bar_refresh_rate=0
    )
    logger.info("======= Training =======")
    trainer.fit(litModel, dataset)
    logger.info("======= Testing =======")
    trainer.test(litModel, datamodule=dataset)
    wandb.finish()




if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
