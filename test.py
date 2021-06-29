
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
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra

from datasets.dataset import Dataset
from datasets.synthetic_data.spiral import Spiral
from models.dnn import SimpleNet
from experiments.litExperiment import LitExperiment
from experiments.experiment_multiclass import ExperimentMulti
from utils.utils import hammingDistance
from utils.get_signatures import get_signatures

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

    # W&B INIT
    config = edict()
    for value in CFG.values():
        config.update(value)
    run = wandb.init(project=CFG.EXPERIMENT.wandb_project, job_type="test", config=config, name=CFG.EXPERIMENT.name)   
    
    # GET MODEL FROM W&B ARTIFACT
    api = wandb.Api()
    artifact = api.artifact(f'{CFG.EXPERIMENT.wandb_project}/{CFG.MODEL.name}-{CFG.EXPERIMENT.name}:seed{CFG.Logging.seed}')
    model_dir = artifact.checkout()
    model = torch.load(model_dir + f'/{CFG.MODEL.name}-{CFG.EXPERIMENT.name}.pt')
    model.eval()

    # GET DATA
    for traj_type in ['same_class', 'diff_class']:
        trajectory, traj_len = Spiral.make_trajectory(type=traj_type)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, sigs, _ = get_signatures(torch.tensor(trajectory).float().to(device), model)
        h_distance = hammingDistance(sigs.float(), device=device)
        avg_trans = torch.diag(h_distance[:, 1:])
        data = [[i, t] for i, t in enumerate(avg_trans)]
        table = wandb.Table(data=data, columns=['x', 'y'])
        wandb.log({f"transitions-{traj_type}" : wandb.plot.line(table, "x", "y",
            title="Custom Y vs X Line Plot")})

    # # get datasets
    # dataset = Dataset(CFG.DATASET)
    # dataset.plot('./')

    # # build model
    # model = SimpleNet(CFG.MODEL)
    # logger.info("[Model] Building model {} out dim: {}".format(CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    # if CFG.EXPERIMENT.use_gpu:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = model.to(device=device)

    # experiment = Experiment(model, dataset, CFG, plot_sig=True)
    # experiment.load_model(Path(CFG.EXPERIMENT.resume_checkpoints))
    # experiment.testing()
    # logger.info("======= test done =======")

if __name__ == '__main__':
    main()

