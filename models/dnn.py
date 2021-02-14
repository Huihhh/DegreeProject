from torch import nn
import torch

ACT_METHOD = {
    'relu': nn.ReLU(),
    'leakyRelU': nn.LeakyReLU()
}

INIT_METHOD = {
    'normal': torch.nn.init.normal_,
    'he_normal': lambda x: torch.nn.init.kaiming_normal_(x, nonlinearity='relu'),
    'xavier_normal': torch.nn.init.xavier_normal_,
    'zeros': torch.nn.init.zeros_,
    'ones': torch.nn.init.ones_
}

class SimpleNet(nn.Sequential):
    # h_nodes: include input_dim
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.layers = []
        self.h_nodes = cfg.h_nodes
        self.out_dim = cfg.out_dim
        self.use_bn = cfg.use_bn

        for i in range(len(self.h_nodes) - 1):
            fc = nn.Linear(self.h_nodes[i], self.h_nodes[i+1])
            torch.random.manual_seed(i)
            INIT_METHOD[self.cfg.fc_winit](fc.weight)
            INIT_METHOD[self.cfg.fc_binit](fc.bias)
            ac = ACT_METHOD[self.cfg.activation]
            if self.use_bn:
                bn = nn.BatchNorm1d(self.h_nodes[i+1])
                INIT_METHOD[self.cfg.bn_winit](bn.weight)
                INIT_METHOD[self.cfg.bn_binit](bn.bias)
                self.layers.append(nn.Sequential(fc, bn, ac))
            else:
                self.layers.append(nn.Sequential(fc, ac))

        predict = nn.Linear(self.h_nodes[-1], self.out_dim)
        self.layers.append(predict)
        super().__init__(*self.layers)


    def __str__(self) -> str:
        return super().__str__() + '\ntorch.sigmoid'

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
    import os, sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    from utils.get_signatures import get_signatures

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG.MODEL), '\n')
        net = SimpleNet(CFG.MODEL)
        h = 0.01
        xx, yy = np.meshgrid(np.arange(-1, 1, h),
                            np.arange(-1, 1, h))
        grid_points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        sigs_grid, net_out, _ = get_signatures(torch.tensor(grid_points).float(), net)

        print(net)

    main()

