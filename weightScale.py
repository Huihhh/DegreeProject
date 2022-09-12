import os
import logging
from typing import Any, Callable
import torch
from torch import nn

import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from datasets.dataset import Dataset
from nn_models import *
from utils import set_random_seed, flat_omegadict, get_signatures
from utils.init_methods import normal_custom

def get_var(model: 'nn.Module') -> float:
    '''
    calculate the variance of all the model weights (bias excluded)

    Parameter
    ----------
    * model: pytorch nn.Module
    '''
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.extend(param.reshape(-1).tolist())
    
    return np.array(weights).var()


class WeightScaleOnLR:
    def __init__(self, model: 'nn.Module', dataset: Any, CFG: DictConfig) -> None:
        '''
        Compare the number of linear regions among networks initialized with increasing std. (no training involved)
        nn.init.normal_ is used.

        parameter
        ------------
        * model: pytorch nn.Module
        * dataset: must has property 'grid_data'
        * CFG: config for wandb logging
        '''
        self.model = model
        self.grid_points, self.grid_labels = dataset.grid_data
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_methods = {
            'custom': normal_custom, 
            'he_normal': torch.nn.init.kaiming_normal_,
            'he_uniform': torch.nn.init.kaiming_uniform_,
            'xavier_normal': torch.nn.init.xavier_normal_,
            'xavier_unifom': torch.nn.init.xavier_uniform_,
        }

        config = flat_omegadict(CFG)
        wandb.init(project=CFG.EXPERIMENT.wandb_project, name=CFG.EXPERIMENT.name, config=config)
        self.xname = CFG.EXPERIMENT.comp_scale
        self.comp_method = CFG.EXPERIMENT.comp_method

    def reinit_model(self, init_method: Callable, **kwargs) -> None:
        def reinit_net(net):
            if type(net) is nn.Sequential or issubclass(type(net), nn.Sequential):
                for layer in net:
                    reinit_net(layer)
            elif isinstance(net, nn.Linear):
                if self.xname == 'weight':
                    init_method(net.weight, **kwargs)
                else:
                    init_method(net.bias, **kwargs)

        reinit_net(self.model)
    
    def run(self):
        if self.comp_method:
            for name, method in self.init_methods.items():
                self.reinit_model(method)
                scale = get_var(self.model)
                _, grid_sigs, _ = get_signatures(torch.tensor(self.grid_points).float().to(self.device), self.model, self.device)
                num_lr = len(torch.unique(grid_sigs, dim=0)) 
                wandb.log({'var': scale, '#Linear regions': num_lr})
        else:
            for gain in np.arange(0.001, 20, 0.01):
                self.reinit_model(nn.init.normal_, std=gain)
                _, grid_sigs, _ = get_signatures(torch.tensor(self.grid_points).float().to(self.device), self.model, self.device)
                num_lr = len(torch.unique(grid_sigs, dim=0)) 
                wandb.log({'weight_std': gain, '#Linear regions': num_lr})
        


    
@hydra.main(config_path='./config', config_name='weightScale')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))

    # # For reproducibility, set random seed
    set_random_seed(CFG.seed)

    if (device_n := CFG.CUDA_VISIBLE_DEVICES) is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_n)
        print('CUDA_VISIBLE_DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'])

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


    
    