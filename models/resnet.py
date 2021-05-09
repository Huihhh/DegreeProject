from torch import nn
from torch.nn.modules import Module
import torchvision.models as models
import torch
from utils.init_methods import *

ACT_METHOD = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}


class ResNet(Module):
    def __init__(self,
                 h_nodes,
                 out_dim,
                 activation,
                 use_bn,
                 fc_winit,
                 fc_binit,
                 bn_winit,
                 bn_binit,
                 seed=0,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)

        # *** FC layers ***
        h_nodes = [resnet18.fc.in_features] + list(h_nodes)
        self.use_bn = use_bn
        self.n_neurons = sum(h_nodes) + out_dim
        self.layers = []
        for i in range(len(h_nodes) - 1):
            torch.random.manual_seed(i + seed)
            fc = nn.Linear(h_nodes[i], h_nodes[i + 1])
            if fc_winit.name != 'default':  # TODO: more elegant way
                eval(fc_winit.func)(fc.weight, **fc_winit.params)
            if fc_binit.name != 'default':
                eval(fc_binit.func)(fc.bias, **fc_binit.params)
            ac = ACT_METHOD[activation]
            if use_bn:
                bn = nn.BatchNorm1d(h_nodes[i + 1])
                if bn_winit.name != 'default':
                    eval(bn_winit.func)(bn.weight, **bn_winit.params)
                if bn_binit.name != 'default':
                    eval(bn_binit.func)(bn.bias, **bn_binit.params)
                self.layers.append(nn.Sequential(fc, bn, ac))
            else:
                self.layers.append(nn.Sequential(fc, ac))

        predict = nn.Linear(h_nodes[-1], out_dim)
        if fc_winit.name != 'default':
            eval(fc_winit.func)(predict.weight, **fc_winit.params)
        if fc_binit.name != 'default':
            eval(fc_binit.func)(predict.bias, **fc_binit.params)
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
        net = ResNet(**CFG.MODEL)
        print(net)

    main()
