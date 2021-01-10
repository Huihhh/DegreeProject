from torch import nn

class SimpleNet(nn.Module):
    # h_nodes: include input_dim
    def __init__(self, h_nodes=[2, 10, 10], num_class=2) -> None:
        super().__init__()
        self.fcs = []
        self.bns = []
        self.acs = []

        for i in range(len(h_nodes) - 1):
            bn = nn.BatchNorm1d(h_nodes[i])
            self.bns.append(bn)
            setattr(self, 'bn%i'%i, bn)
            fc = nn.Linear(h_nodes[i], h_nodes[i+1])
            self.fcs.append(fc)
            setattr(self, 'fc%i'%i, fc)
            ac = nn.ReLU()
            self.acs.append(ac)
            setattr(self, 'ac%i'%i, ac)

        self.predict = nn.Linear(h_nodes[-1], num_class)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i in range(len(self.fcs)):
            x = self.bns[i](x)
            x = self.fcs[i](x)
            x = self.acs[i](x)

        out = self.softmax(self.predict(x))
        return out

if __name__ == "__main__":
    net = SimpleNet()
    print(net)
