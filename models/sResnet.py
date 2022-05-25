import os
import sys
import re
from collections import OrderedDict
from torch import nn
from torch.nn.modules import Module
import torchvision.models as models
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.init_methods import *

ACT_METHOD = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SResNet(Module):
    def __init__(self,
                 h_nodes: list[int],
                 out_dim: int,
                 activation: str,
                 dropout: float,
                 fc_winit: dict,
                 fc_binit: dict,
                 use_bn: bool=False,
                 seed: int=0,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.resnet = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(32)),
            ('ac', nn.ReLU()),
            ('pool', nn.MaxPool2d(3, stride=2, padding=1)),
            ('res', BasicBlock(32, 32)),
            ('pool', nn.AdaptiveAvgPool2d((4, 4)))
        ]
        ))

        # *** FC layers ***
        h_nodes = [512] + list(h_nodes)
        self.use_bn = use_bn
        self.n_neurons = sum(h_nodes)
        self.layers = []
        for i in range(len(h_nodes) - 1):
            s = nn.Sequential()
            torch.random.manual_seed(i + seed)
            s.add_module('fc', nn.Linear(h_nodes[i], h_nodes[i + 1]))
            s.add_module('ac', ACT_METHOD[activation])
            if use_bn:
                s.add_module('bn', nn.BatchNorm1d(h_nodes[i + 1]))
            if dropout != 0:
                s.add_module('dropout', nn.Dropout(p=dropout))
            self.layers.append(s)
        
        # predict layer
        self.layers.append(nn.Linear(h_nodes[-1], out_dim))
        self.fcs = torch.nn.Sequential(*self.layers)

        if fc_winit.name != 'default' or fc_binit.name != 'default':
            self.reset_parameters(fc_winit, fc_binit)

        # *** ResNet ***
        print(self.resnet)   

    def reset_parameters(self, winit: dict, binit: dict, seed: int=0) -> None:
        '''
        reinit the model's weights and bias (batchnorm weights excluded)

        Parameter
        ---------
        * winit: dict, init method for weights
        * binit: dict, init method for bias
        * seed: random seed
        '''
        p1 = re.compile(r'^((?!bn).)*weight') #find conv weight
        p2 = re.compile(r'^((?!bn).)*bias') # excluding bn bias
        for i, (name, param) in enumerate(self.named_parameters()):
            torch.random.manual_seed(i + seed)
            if winit.name != 'default' and p1.search(name):
                eval(winit.func)(param, **winit.params)
                continue
            if binit.name != 'default' and p2.search(param):
                eval(binit.func)(param, **binit.params)

    def forward(self, x):
        x = self.resnet(x).reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
    
    def feature_forward(self, x):
        features = x = self.resnet(x).reshape(x.shape[0], -1)
        pre_ac = []
        for net in self.fcs[:-1]:
            x = net.fc(x)
            pre_ac.append(x)
            x = net.ac(x) #TODO: if use bn
        x = self.fcs[-1](x)
        pre_ac = torch.cat(pre_ac, dim=1)
        return x, pre_ac, features

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
        net = SResNet(**CFG.MODEL, dropout=0)
        print(net)
        inputs = torch.randn((1, 3, 64, 64))
        outputs = net(inputs)
        print(outputs.shape)

    main()
