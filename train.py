from experiments.experiment_multiclass import ExperimentMulti
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omegaconf import DictConfig, OmegaConf
import hydra
import ignite
import logging

from datasets.dataset import Dataset
from models import *
from experiments.litExperiment import LitExperiment
 
@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))

    # # For reproducibility, set random seed
    if CFG.Logging.seed == 'None':
        CFG.Logging.seed = random.randint(1, 10000)
    random.seed(CFG.Logging.seed)
    np.random.seed(CFG.Logging.seed)
    torch.manual_seed(CFG.Logging.seed)
    torch.cuda.manual_seed_all(CFG.Logging.seed)
    ignite.utils.manual_seed(CFG.Logging.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if CFG.DATASET.name == 'eurosat':
        model = ResNet(**CFG.MODEL)
        dataset = Dataset(model.resnet18, **CFG.DATASET)
        input_dim = 512
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

    if CFG.MODEL.name == 'resnet':
        experiment = ExperimentMulti(model, dataset, CFG)
    else:
        experiment = LitExperiment(model, dataset, CFG)
    logger.info("======= Training =======")
    experiment.run()


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
