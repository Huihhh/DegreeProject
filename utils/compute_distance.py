import torch
import torch.nn as nn
import numpy as np

def compute_distance(data, net, min_distance = None):
    if min_distance is None:
        min_distance = torch.ones((data.shape[0])) * np.inf
    if type(net) is nn.Sequential or issubclass(type(net), nn.Sequential):
        # Sequential: go over each layer and record activation patterns
        for op in net:
            data, min_distance = compute_distance(data, op, min_distance)
        return data, min_distance
    elif (type(net) is nn.Linear):
        # #TODO: the numerator is z(x) - b in the paper 'Complexity of Linear Regions in Deep Networks'
        net_out = net(data)
        min_d, _ = torch.min(torch.abs(net_out) / torch.norm(net.weight), 1)
        min_distance = torch.min(min_distance, min_d)
        return net_out, min_distance
    else:
        return net(data), min_distance