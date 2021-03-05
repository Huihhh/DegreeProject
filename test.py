
""" 
Get test accuracy of checkpoints
run: 
python test.py hydra.run.dir="outputs/output_folder"

example: 
python test.py hydra.run.dir="outputs/circles_fill_xu_ru_seed0_2021-03-03_10-47-32"
"""

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random, yaml
from easydict import EasyDict as edict
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

from datasets.syntheticData import Dataset
from models.dnn import SimpleNet
from experiments.experiment import Experiment




@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)

    with open('./.hydra/config.yaml', 'r') as file:
        try:
            config_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    TRAIN_CFG = edict(config_file)
    CFG.MODEL.h_nodes = TRAIN_CFG.MODEL.h_nodes
    logger.info(OmegaConf.to_yaml(CFG))

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
    dataset.plot('./')

    # build model
    model = SimpleNet(CFG.MODEL)
    logger.info("[Model] Building model {} out dim: {}".format(CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    if CFG.EXPERIMENT.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)

    experiment = Experiment(model, dataset, CFG, plot_sig=True)
    experiment.load_model(Path(CFG.EXPERIMENT.resume_checkpoints))
    experiment.testing()
    logger.info("======= test done =======")

if __name__ == '__main__':
    main()

