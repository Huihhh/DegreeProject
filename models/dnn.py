from torch import nn
import torch

class SimpleNet(nn.Sequential):
    # h_nodes: include input_dim
    def __init__(self, h_nodes=[2, 16, 10], out_dim=1) -> None:
        self.layers = []

        for i in range(len(h_nodes) - 1):
            # bn = nn.BatchNorm1d(h_nodes[i])
            # setattr(self, 'bn%i'%i, bn)
            fc = nn.Linear(h_nodes[i], h_nodes[i+1])
            nn.init.xavier_normal_(fc.weight)
            nn.init.constant_(fc.bias, 0)
            # setattr(self, 'fc%i'%i, fc)
            ac = nn.ReLU()
            bn = nn.BatchNorm1d(h_nodes[i+1])
            # setattr(self, 'ac%i'%i, ac)
            self.layers.append(nn.Sequential(fc, ac, bn))

        predict = nn.Linear(h_nodes[-1], out_dim)
        nn.init.xavier_normal_(predict.weight)
        nn.init.constant_(predict.bias, 0)
        # softmax = nn.Softmax(dim=1)
        self.layers.append(predict)
        # self.layers.append(softmax)
        super().__init__(*self.layers)
    
    def forward(self, input):
        x = super().forward(input)
        return torch.sigmoid(x)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os
    import sys
    sys.path.append(os.getcwd())

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG.MODEL), '\n')
        net = SimpleNet(**CFG.MODEL)
        print(issubclass(type(net), nn.Sequential))
        print('=========')
        for layer in net:
            print(layer)
            if isinstance(layer, nn.Linear):
                print(layer.bias)


    main()

