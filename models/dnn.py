from torch import nn
import torch
import numpy as np

from utils.init_methods import *


ACT_METHOD = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}


class SimpleNet(nn.Sequential):
    def __init__(self, cfg, input_dim) -> None:
        self.cfg = cfg
        self.layers = []
        self.h_nodes = [input_dim] + list(cfg.h_nodes)
        self.n_neurons = sum(self.h_nodes) + 1
        self.out_dim = cfg.out_dim
        self.use_bn = cfg.use_bn

        for i in range(len(self.h_nodes) - 1):
            torch.random.manual_seed(i+self.cfg.seed)
            fc = nn.Linear(self.h_nodes[i], self.h_nodes[i+1])
            if cfg.fc_winit.name != 'default': #TODO: more elegant way
                eval(cfg.fc_winit.func)(fc.weight, **cfg.fc_winit.params)
            if cfg.fc_binit.name != 'default':
                eval(cfg.fc_binit.func)(fc.bias, **cfg.fc_binit.params)
            ac = ACT_METHOD[self.cfg.activation]
            if self.use_bn:
                bn = nn.BatchNorm1d(self.h_nodes[i+1])
                if cfg.bn_winit.name != 'default':
                    eval(cfg.bn_winit.func)(bn.weight, **cfg.bn_winit.params)
                if cfg.bn_binit.name != 'default':
                    eval(cfg.bn_binit.func)(bn.bias, **cfg.bn_binit.params)
                self.layers.append(nn.Sequential(fc, bn, ac))
            else:
                self.layers.append(nn.Sequential(fc, ac))

        predict = nn.Linear(self.h_nodes[-1], self.out_dim)
        if cfg.fc_winit.name != 'default':
            eval(cfg.fc_winit.func)(predict.weight, **cfg.fc_winit.params)
        if cfg.fc_binit.name != 'default':
            eval(cfg.fc_binit.func)(predict.bias, **cfg.fc_binit.params)
        self.layers.append(predict)
        super().__init__(*self.layers)

    def __str__(self) -> str:
        return super().__str__() + '\ntorch.sigmoid'

    # def param_init(self):
    #     if self.cfg.
    #     init_method =
    #     for layer in self.layers:
    #         if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):

    def forward(self, input):
        x = super().forward(input)
        return torch.sigmoid(x)


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
        net = SimpleNet(CFG.MODEL)
        h = 0.01
        xx, yy = np.meshgrid(np.arange(-1, 1, h),
                             np.arange(-1, 1, h))
        grid_points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        out, min_distance = compute_distance(torch.tensor([[0.1, 0.2], [0.5, 0.7]]), net)
        sigs_grid, net_out, _ = get_signatures(
            torch.tensor(grid_points).float(), net)

        print(net)

    main()
