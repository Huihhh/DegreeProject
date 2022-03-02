from torch import nn
from torch.nn.modules import Module
import torchvision.models as models
import torch
import os
import sys
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
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            BasicBlock(32, 32),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # *** FC layers ***
        h_nodes = [512] + list(h_nodes)
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
        print(self.resnet)     

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
        net = SResNet(**CFG.MODEL)
        print(net)
        inputs = torch.randn((1, 3, 64, 64))
        outputs = net(inputs)
        print(outputs.shape)

    main()
