import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

gpu = True
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')


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
    if type(net) is nn.Sequential or issubclass(type(net), nn.Sequential):
        # Sequential: go over each layer and record activation patterns
        signatures = []
        sizes = []
        for op in net:
            data, op_sig, sz = get_signatures(data, op)
            signatures.append(op_sig)
            sizes.append(sz)
        return data, torch.cat(signatures, dim=1), sizes
    elif (type(net) is nn.Linear) or (type(net) is nn.BatchNorm1d):
        # Linear: no non-linearities occur
        return net(data), torch.zeros(data.shape[0], 0, device=device, dtype=torch.int8), 0
    elif type(net) is nn.ReLU or type(net) is nn.LeakyReLU:
        # ReLu: each neuron creates a non-linearity
        signatures = (data > 0).type(torch.int8)
        # print(signatures.shape)
        return net(data), signatures, signatures.shape[1]
    elif type(net) is nn.Dropout: #TODO: currently use net.eval() when plot linear regions
        return data, torch.zeros(data.shape[0], 0, device=device, dtype=torch.int8), 0
    else:
        raise Exception('Unknown operation: ' + str(net))

COLOR_Pastel2 = ('rgb(179,226,205)', 'rgb(253,205,172)', 'rgb(203,213,232)', 'rgb(244,202,228)', 'rgb(230,245,201)', 'rgb(255,242,174)', 'rgb(241,226,204)', 'rgb(204,204,204)')
def visualize_signatures(grid_sigs,grid_labels, grid_points): # ? is confidence map meaningfull to visualize?
    # * signatures:activation patterns
    # * grid_sigs: signatures of grid points
    xx, yy = grid_points[:, 0], grid_points[:, 1]
    region_sigs = list(np.unique(grid_sigs)) # signatures of regions
    total_regions = {}
    total_regions['density'] = len(region_sigs)
    region_ids = np.random.permutation(total_regions['density'])

    sigs_grid_dict = dict(zip(region_sigs, region_ids))
    base_color_labels = np.array([sigs_grid_dict[sig] for sig in grid_sigs])
    base_color_labels = base_color_labels.reshape(grid_labels.shape).T

    grid_labels_vec = grid_labels.reshape(-1)
    color_labels = np.zeros(grid_labels_vec.shape)
    for i, key in enumerate(region_sigs): # loop through key, for each region, calculate the ratio of positive samples
        idx = np.where(grid_sigs == key)
        region_labels = grid_labels_vec[idx]
        ratio = sum(region_labels) / region_labels.size 
        color_labels[idx] = ratio

    color_labels = color_labels.reshape(grid_labels.shape).T
    # color_labels[np.where(color_labels>=0.2)] = 1
    # color_labels[np.where(color_labels<=-0.2)] = -1
    # color_labels[np.where((color_labels>-0.2) & (color_labels<0.2))] = 0
    fig = make_subplots(rows=1, cols=1)
    layer1 = go.Heatmap(z=base_color_labels,  opacity=0.6, colorscale=COLOR_Pastel2)
    layer2 = go.Heatmap(z=color_labels, opacity=0.6, colorscale=['rgb(255,0,0)', 'rgb(255,255,255)', 'rgb(0,0,250)'])
    fig.add_trace(layer1, 1, 1)
    fig.add_trace(layer2, 1, 1)
    fig.update_layout(
        yaxis = dict(scaleanchor = 'x'), # set aspect ratio to 1
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    ) 
    fig.show()
    return fig


def compute_distance(data, net, min_distance=None):
        ##
    # Compute the distance of a point to its nearest linear region.
    # Input:
    #  * data: Nx? matrix of N data points
    #  * net: pytorch network,
    # Output:
    #  * data_out: net(data)
    #  * min_distance: Nx(Q1+Q2+..+Ql) binary matrix,
    ##
    if min_distance is None:
        min_distance = torch.ones(
            (data.shape[0]),
            device='cuda' if torch.cuda.is_available() else 'cpu') * np.inf
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

