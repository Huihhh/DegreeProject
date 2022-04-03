from matplotlib.pyplot import subplot
import torch
import torch.nn as nn
from utils import get_signatures
import numpy as np
import wandb
import os

from omegaconf import DictConfig, OmegaConf
import hydra
import logging

from datasets.dataset import Dataset
from models import *
from utils import set_random_seed, flat_omegadict

#  * set cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class WeightScaleOnLR:
    def __init__(self, model, dataset, CFG) -> None:
        self.model = model
        self.dataset = dataset # TODO: remove later
        self.grid_points, self.grid_labels = dataset.grid_data
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wandb.init(project=CFG.EXPERIMENT.wandb_project, name=CFG.EXPERIMENT.name, config=flat_omegadict(CFG))
        self.xname = CFG.EXPERIMENT.comp_scale


    # * STEP1. init model
    def reinit_model(self, std):
        def reinit_net(net):
            if type(net) is nn.Sequential or issubclass(type(net), nn.Sequential):
                for layer in net:
                    reinit_net(layer)
            elif isinstance(net, nn.Linear):
                if self.xname == 'weight':
                    nn.init.normal_(net.weight, std=std)
                else:
                    nn.init.normal_(net.bias, std=std)

        reinit_net(self.model)

    def run(self):
        for gain in np.arange(1, 20, 0.01): #0.001
            self.reinit_model(gain)
            # * STEP2. get signatures and count
            _, grid_sigs, _ = get_signatures(torch.tensor(self.grid_points).float(), self.model) # TODO: bring .to(self.device) back later
            num_lr = len(torch.unique(grid_sigs, dim=0)) 
            wandb.log({'weight_std': gain, '#Linear regions': num_lr})
    
@hydra.main(config_path='./config', config_name='weightScale')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))

    # # For reproducibility, set random seed
    set_random_seed(CFG.seed)

    # * make sure right data and model have been chosen
    assert CFG.DATASET.name != 'eurosat', '! eurosat is not suitable for this experiment. Availabel datasets: spiral, circles, moons'
    assert CFG.MODEL.name not in ['resnet', 'sResnet'], f'! {CFG.MODEL.name} is not for this experiment, please use  shallow_nn'
    # get datasets
    dataset = Dataset(seed=CFG.seed, **CFG.DATASET)
    input_dim = dataset.trainset[0][0].shape[0]

    # build model
    model = MODEL[CFG.MODEL.name](input_dim=input_dim, seed=CFG.seed, **CFG.MODEL)
    logger.info("[Model] Building model -- input dim: {}, hidden nodes: {}, out dim: {}"
                                .format(input_dim, CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    if CFG.EXPERIMENT.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)


    experiment = WeightScaleOnLR(model, dataset, CFG)
    experiment.run()


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()


    
    