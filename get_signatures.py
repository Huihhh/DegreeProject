import torch
import torch.nn as nn

gpu = True
device = torch.device('cuda' if gpu else 'cpu')


##
# Compute activation pattens of the provided data.
# Input:
#  * data: Nx? matrix of N data points
#  * net: pytorch network,
#    supported layers: Sequential, Linear, Relu, LeakyReLu
# Output:
#  * data_out: net(data)
#  * signatures: Nx(Q1+Q2+..+Ql) binary matrix,
#    where l is the number of layers
#    and Qi is the number of nonlinearities on the ith layer
#  * sizes: all Qi with the preserved structure of the network
##
def get_signatures(data, net):
    if type(net) is nn.Sequential:
        # Sequential: go over each layer and record activation patterns
        signatures = []
        sizes = []
        for op in net:
            data, op_sig, sz = get_signatures(data, op)
            signatures.append(op_sig)
            sizes.append(sz)
        return data, torch.cat(signatures, dim=1), sizes
    elif type(net) is nn.Linear:
        # Linear: no non-linearities occur
        return net(data), torch.zeros(data.shape[0], 0, device=device, dtype=torch.int8), 0
    elif type(net) is nn.ReLU or type(net) is nn.LeakyReLU:
        # ReLu: each neuron creates a non-linearity
        signatures = (data > 0).type(torch.int8)
        return net(data), signatures, signatures.shape[1]
    else:
        raise Exception('Unknown operation: ' + str(net))
