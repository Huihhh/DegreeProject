from torch import nn

class SimpleNet(nn.Sequential):
    # h_nodes: include input_dim
    def __init__(self, h_nodes=[2, 16, 10], num_class=2) -> None:
        self.layers = []

        for i in range(len(h_nodes) - 1):
            # bn = nn.BatchNorm1d(h_nodes[i])
            # setattr(self, 'bn%i'%i, bn)
            fc = nn.Linear(h_nodes[i], h_nodes[i+1])
            # setattr(self, 'fc%i'%i, fc)
            ac = nn.ReLU()
            # setattr(self, 'ac%i'%i, ac)
            self.layers.append(nn.Sequential( fc, ac))

        predict = nn.Linear(h_nodes[-1], num_class)
        # softmax = nn.Softmax(dim=1)
        self.layers.append(predict)
        # self.layers.append(softmax)
        super().__init__(*self.layers)


if __name__ == "__main__":
    net = SimpleNet()
    print(issubclass(type(net), nn.Sequential))
    print('=========')
    for layer in net:
        print(layer)
