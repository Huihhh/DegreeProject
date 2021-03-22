from torch import nn
import torch
import numpy as np


def range_uniform(x, a=0.8, b=1.2):
    n = x.shape[0] // 2
    x1 = torch.nn.init.uniform_(x[:n, ], a, b)
    x2 = torch.nn.init.uniform_(x[n:, ], -b, -a)
    return torch.cat([x1, x2], 0)
    # signs = torch.randint(0,2, x.shape)
    # signs[torch.where(signs==0)] = -1
    # return signs * torch.nn.init.uniform_(x, a, b)


def tanh(w, a=-np.pi, b=np.pi):
    n, m = w.shape
    ww = np.zeros((n, m))
    for i in range(m):
        wn = np.tanh(np.linspace(a, b, n))
        ww[:, i] = wn
    w = torch.tensor(ww, requires_grad=True,
                     device="cuda" if torch.cuda.is_available() else 'cpu')
    # w.to("cuda" if torch.cuda.is_available() else 'cpu')
    # w.requires_grad = True
    return w


def cos(w, a=-np.pi, b=np.pi):
    n, = w.shape
    wn = np.cos(np.linspace(a, b, n))
    w = torch.tensor(wn, requires_grad=True,
                     device="cuda" if torch.cuda.is_available() else 'cpu')
    # w.to("cuda" if torch.cuda.is_available() else 'cpu')
    # w.requires_grad = True
    return w


ACT_METHOD = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}


class SimpleNet(nn.Sequential):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.layers = []
        self.h_nodes = [eval(cfg.input_dim)] + list(cfg.h_nodes)
        self.out_dim = cfg.out_dim
        self.use_bn = cfg.use_bn

        for i in range(len(self.h_nodes) - 1):
            torch.random.manual_seed(i+self.cfg.seed)
            fc = nn.Linear(self.h_nodes[i], self.h_nodes[i+1])
            if cfg.fc_winit.name != 'default': #TODO: more elegant way
                eval(cfg.fc_winit.func)(fc.weight, **cfg.fc_winit.params)
                eval(cfg.fc_binit.func)(fc.bias, **cfg.fc_binit.params)
            ac = ACT_METHOD[self.cfg.activation]
            if self.use_bn:
                bn = nn.BatchNorm1d(self.h_nodes[i+1])
                if cfg.fc_winit.name != 'default':
                    eval(cfg.bn_winit.func)(bn.weight, **cfg.bn_winit.params)
                    eval(cfg.bn_binit.func)(bn.bias, **cfg.bn_binit.params)
                self.layers.append(nn.Sequential(fc, bn, ac))
            else:
                self.layers.append(nn.Sequential(fc, ac))

        predict = nn.Linear(self.h_nodes[-1], self.out_dim)
        if cfg.fc_winit.name != 'default':
            eval(cfg.fc_winit.func)(predict.weight, **cfg.fc_winit.params)
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

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG.MODEL), '\n')
        net = SimpleNet(CFG.MODEL)
        h = 0.01
        xx, yy = np.meshgrid(np.arange(-1, 1, h),
                             np.arange(-1, 1, h))
        grid_points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        sigs_grid, net_out, _ = get_signatures(
            torch.tensor(grid_points).float(), net)

        print(net)

    main()
