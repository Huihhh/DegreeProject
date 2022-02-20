import torch
import wandb
import torch.nn as nn
from utils import get_signatures
from easydict import EasyDict as edict

import random
import numpy as np
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
import math

from datasets.dataset import Dataset
from models import *
from utils import set_random_seed


class WeightScaleOnLR:
    def __init__(self, model, dataset, CFG) -> None:
        self.model = model
        self.grid_points, self.grid_labels = dataset.grid_data
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = edict()
        for value in CFG.values():
            config.update(value)
        wandb.init(project=CFG.EXPERIMENT.wandb_project, name=CFG.EXPERIMENT.name, config=config)
        self.xname = CFG.EXPERIMENT.comp_scale
        # self.init_method = CFG.MODEL.fc_winit if self.xname == 'weight' else CFG.MODEL.fc_binit

        # self.init_method = eval(self.init_method.func)


    # * STEP1. init model & calculate std
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
                # fan_in = net.weight.shape[1]
                # bound = std / math.sqrt(fan_in) if fan_in > 0 else 0
                # nn.init.uniform_(net.bias, -bound, bound)
        
        reinit_net(self.model)
        weights = []
        for name, param in self.model.named_parameters(): # ? include bias?
            if self.xname in name:
                weights.extend(param.reshape(-1).tolist())
        return np.array(weights).std()

    def run(self):
        for gain in np.arange(0.001, 20, 0.01):
            scale = self.reinit_model(gain)
            # * STEP2. get signatures and count
            _, sigs_grid, _ = get_signatures(torch.tensor(self.grid_points).float().to(self.device), self.model)
            num_lr = len(set([''.join(str(x) for x in s.tolist()) for s in sigs_grid])) 
            wandb.log({'weight_std': gain, '#Linear regions': num_lr})

        


    
@hydra.main(config_path='./config', config_name='weightScale')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))

    # # For reproducibility, set random seed
    set_random_seed(CFG.Logging.seed)

    # * make sure right data and model have been chosen
    assert CFG.DATASET.name != 'eurosat', '! eurosat is not suitable for this experiment. Availabel datasets: spiral, circles, moons'
    assert CFG.MODEL.name not in ['resnet', 'sResnet'], f'! {CFG.MODEL.name} is not for this experiment, please use  shallow_nn'
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


    experiment = WeightScaleOnLR(model, dataset, CFG)
    experiment.run()


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()


    
    