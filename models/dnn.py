from torch import nn
import torch
import numpy as np

from utils.init_methods import *

ACT_METHOD = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU()}


class SimpleNet(nn.Sequential):
    def __init__(self,
                 input_dim,
                 h_nodes,
                 out_dim,
                 activation,
                 fc_winit,
                 fc_binit,
                 bn_winit,
                 bn_binit,
                 use_bn=False,
                 seed=0, **kwargs) -> None:
        self.use_bn = use_bn
        h_nodes = [input_dim] + list(h_nodes)
        self.n_neurons = sum(h_nodes)

        self.layers = []
        for i in range(len(h_nodes) - 1):
            s = nn.Sequential()
            torch.random.manual_seed(i + seed)
            fc = nn.Linear(h_nodes[i], h_nodes[i + 1])
            if fc_winit.name != 'default':  #TODO: more elegant way
                eval(fc_winit.func)(fc.weight, **fc_winit.params)
            if fc_binit.name != 'default':
                eval(fc_binit.func)(fc.bias, **fc_binit.params)
            s.add_module('fc', fc)
            ac = ACT_METHOD[activation]
            s.add_module('ac', ac)
            if self.use_bn:
                bn = nn.BatchNorm1d(h_nodes[i + 1])
                if bn_winit.name != 'default':
                    eval(bn_winit.func)(bn.weight, **bn_winit.params)
                if bn_binit.name != 'default':
                    eval(bn_binit.func)(bn.bias, **bn_binit.params)
                s.add_module('bn', bn)            
            self.layers.append(s)

        predict = nn.Linear(h_nodes[-1], out_dim)
        if fc_winit.name != 'default':
            eval(fc_winit.func)(predict.weight, **fc_winit.params)
        if fc_binit.name != 'default':
            eval(fc_binit.func)(predict.bias, **fc_binit.params)
        self.layers.append(predict)
        super().__init__(*self.layers)

    def __str__(self) -> str:
        return super().__str__() + '\ntorch.sigmoid'

    # def forward(self, input):
    #     x = super().forward(input)
    #     return torch.sigmoid(x)
       
    def forward(self, x):
        pre_ac = []
        for net in self.layers[:-1]:
            x = net.fc(x)
            pre_ac.append(x)
            x = net.ac(x) #TODO: if use bn
        x = self.layers[-1](x)
        pre_ac = torch.cat(pre_ac, dim=1)
        return torch.sigmoid(x), pre_ac


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os
    import sys
    import numpy as np
    sys.path.append(os.getcwd())
    print(os.getcwd())
    import os
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    from utils.get_signatures import get_signatures
    from utils.compute_distance import compute_distance

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG.MODEL), '\n')
        net = SimpleNet(**CFG.MODEL)
        h = 0.01
        xx, yy = np.meshgrid(np.arange(-1, 1, h), np.arange(-1, 1, h))
        grid_points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        out, min_distance = compute_distance(
            torch.tensor([[0.1, 0.2], [0.5, 0.7]]), net)
        sigs_grid, net_out, _ = get_signatures(
            torch.tensor(grid_points).float(), net)

        print(net)

    main()
