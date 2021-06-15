from torch import nn
from torch.nn.modules import Module
import torchvision.models as models
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.init_methods import *

ACT_METHOD = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}


class SResNet(Module):
    def __init__(self,
                 h_nodes,
                 out_dim,
                 activation,
                 use_bn,
                 dropout,
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
        h_nodes = [64] + list(h_nodes)
        self.use_bn = use_bn
        self.n_neurons = sum(h_nodes)
        self.layers = []
        for i in range(len(h_nodes) - 1):
            s = nn.Sequential()
            torch.random.manual_seed(i + seed)
            s.add_module('fc', nn.Linear(h_nodes[i], h_nodes[i + 1]))
            if fc_winit.name != 'default':  # TODO: more elegant way
                eval(fc_winit.func)(s[0].weight, **fc_winit.params)
            if fc_binit.name != 'default':
                eval(fc_binit.func)(s[0].bias, **fc_binit.params)

            s.add_module('ac', ACT_METHOD[activation])
            if use_bn:
                bn = nn.BatchNorm1d(h_nodes[i + 1])
                if bn_winit.name != 'default':
                    eval(bn_winit.func)(bn.weight, **bn_winit.params)
                if bn_binit.name != 'default':
                    eval(bn_binit.func)(bn.bias, **bn_binit.params)
                s.add_module('bn', bn)
            if dropout != 0:
                dp = nn.Dropout(p=dropout)
                s.add_module('dropout', dp)
            self.layers.append(s)


        predict = nn.Linear(h_nodes[-1], out_dim)
        if fc_winit.name != 'default':
            eval(fc_winit.func)(predict.weight, **fc_winit.params)
        if fc_binit.name != 'default':
            eval(fc_binit.func)(predict.bias, **fc_binit.params)
        self.layers.append(predict)
        self.fcs = torch.nn.Sequential(*self.layers)

        # *** ResNet ***
        self.resnet = torch.nn.Sequential(*(list(resnet18.children())[:5]), nn.AdaptiveAvgPool2d((1, 1)))   
        print(self.resnet)     

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fcs(x)
        return x
    
    def feature_forward(self, x):
        pre_ac = []
        for net in self.fcs[:-1]:
            x = net.fc(x)
            pre_ac.append(x)
            x = net.ac(x) #TODO: if use bn
        x = self.fcs[-1](x)
        pre_ac = torch.cat(pre_ac, dim=1)
        return x, pre_ac

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os
    import sys
    sys.path.append(os.getcwd())
    print(os.getcwd())


    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG.MODEL), '\n')
        net = SResNet(**CFG.MODEL)
        print(net)
        inputs = torch.randn((1, 3, 64, 64))
        outputs = net(inputs)
        print(outputs.shape)

    main()
