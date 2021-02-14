import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omegaconf import DictConfig, OmegaConf
import hydra

from datasets.syntheticData import Dataset
from models.dnn import SimpleNet
from experiments.experiment import Experiment
from utils.utils import setup_default_logging
 
@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    print('==> CONFIG is: \n', OmegaConf.to_yaml(CFG), '\n')

    # initial logging file
    logger = setup_default_logging(CFG, string='Train')
    logger.info(CFG)

    # # For reproducibility, set random seed
    if CFG.Logging.seed == 'None':
        CFG.Logging.seed = random.randint(1, 10000)
    random.seed(CFG.Logging.seed)
    np.random.seed(CFG.Logging.seed)
    torch.manual_seed(CFG.Logging.seed)
    torch.cuda.manual_seed_all(CFG.Logging.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # get datasets
    dataset = Dataset(CFG.DATASET)

    # build model
    model = SimpleNet(CFG.MODEL)
    logger.info("[Model] Building model {} out dim: {}".format(CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    if CFG.EXPERIMENT.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)

    experiment = Experiment(model, dataset, CFG)
    experiment.fitting()
    logger.info("======= Training done =======")
    experiment.testing()
    logger.info("======= Testing done =======")


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
