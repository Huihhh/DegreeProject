from torch import nn
from torch.nn.modules import Module
import torchvision.models as models
import torch
from utils.init_methods import *

ACT_METHOD = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}


class ResNet(Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)
        
        # *** FC layers ***
        self.h_nodes = [resnet18.fc.in_features] + list(cfg.h_nodes)
        self.n_neurons = sum(self.h_nodes) + cfg.out_dim
        self.layers = []
        for i in range(len(self.h_nodes) - 1):
            torch.random.manual_seed(i+cfg.seed)
            fc = nn.Linear(self.h_nodes[i], self.h_nodes[i+1])
            if cfg.fc_winit.name != 'default':  # TODO: more elegant way
                eval(cfg.fc_winit.func)(fc.weight, **cfg.fc_winit.params)
            if cfg.fc_binit.name != 'default':
                eval(cfg.fc_binit.func)(fc.bias, **cfg.fc_binit.params)
            ac = ACT_METHOD[cfg.activation]
            if cfg.use_bn:
                bn = nn.BatchNorm1d(self.h_nodes[i+1])
                if cfg.bn_winit.name != 'default':
                    eval(cfg.bn_winit.func)(bn.weight, **cfg.bn_winit.params)
                if cfg.bn_binit.name != 'default':
                    eval(cfg.bn_binit.func)(bn.bias, **cfg.bn_binit.params)
                self.layers.append(nn.Sequential(fc, bn, ac))
            else:
                self.layers.append(nn.Sequential(fc, ac))

        predict = nn.Linear(self.h_nodes[-1], cfg.out_dim)
        if cfg.fc_winit.name != 'default':
            eval(cfg.fc_winit.func)(predict.weight, **cfg.fc_winit.params)
        if cfg.fc_binit.name != 'default':
            eval(cfg.fc_binit.func)(predict.bias, **cfg.fc_binit.params)
        self.layers.append(predict)
        self.fcs = torch.nn.Sequential(*self.layers)

        # *** ResNet ***
        self.resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-1]))
        # freeze the pretrained model
        for params in self.resnet18.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fcs(x.squeeze())
        return x


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os
    import sys
    sys.path.append(os.getcwd())
    print(os.getcwd())
    import os
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG.MODEL), '\n')
        net = ResNet(CFG.MODEL)
        print(net)
    main()
