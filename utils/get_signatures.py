import torch
import torch.nn as nn
import logging
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def get_signatures(data, net, device):
    '''
    Compute activation pattens of the provided data.

    Perameters
    ----------
    Input:
      * data: Nx? matrix of N data points
      * net: pytorch network,
      supported layers: Sequential, Linear, Relu, LeakyReLu
    Output:
      * data_out: net(data)
      * signatures: Nx(Q1+Q2+..+Ql) binary matrix,
        where l is the number of layers
        and Qi is the number of nonlinearities on the ith layer
      * sizes: all Qi with the preserved structure of the network
    '''
    if type(net) is nn.Sequential or issubclass(type(net), nn.Sequential):
        # Sequential: go over each layer and record activation patterns
        signatures = []
        sizes = []
        for op in net:
            data, op_sig, sz = get_signatures(data, op, device)
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
    elif type(net) is nn.Dropout:  # TODO: currently use net.eval() when plot linear regions
        return data, torch.zeros(data.shape[0], 0, device=device, dtype=torch.int8), 0
    else:
        raise Exception('Unknown operation: ' + str(net))


COLOR_Pastel2 = ('rgb(179,226,205)', 'rgb(253,205,172)', 'rgb(203,213,232)', 'rgb(244,202,228)',
                 'rgb(230,245,201)', 'rgb(255,242,174)', 'rgb(241,226,204)', 'rgb(204,204,204)')
BWR = ['rgb(0,0,255)', 'rgb(255,255,255)', 'rgb(255,0,0)']
def visualize_signatures(region_labels, grid_labels, xvalues, yvalues, data=None, showscale=False, colorbary=0.765, colorbar_len=0.51):
    '''
    Visualize linear regions over the input space. 
    Coloring linear regions by random colors (to differentiate from each other) 
    and gradual colors (to correlate to the classification regions )

    Perameters
    ----------
     * grid_sigs: signatures of grid points
     * region_labels: region labels of grid points, tell each point in which region
     * grid_labels: labels of grid points, can be groud truth or predictions
     * xvalues: unique x values (xaxis range)
     * yvalues: unique y values (yaxis range)
     * data: data points and labels, to stack on the top
    '''
    classified_region_labels = np.zeros(grid_labels.shape)
    region_labels_unique = np.unique(region_labels)
    # loop through key, for each region, calculate the ratio of positive/negative samples
    for i, key in enumerate(region_labels_unique):
        idx = np.where(region_labels == key)
        region_i_labels = grid_labels[idx]
        ratio = sum(region_i_labels) / region_i_labels.size
        classified_region_labels[idx] = ratio

    # classified_region_labels = classified_region_labels.reshape(grid_labels.shape).T
    # classified_region_labels[np.where(classified_region_labels>=0.2)] = 1
    # classified_region_labels[np.where(classified_region_labels<=-0.2)] = -1
    # classified_region_labels[np.where((classified_region_labels>-0.2) & (classified_region_labels<0.2))] = 0
    layers = []
    l1 = go.Heatmap(
        z=classified_region_labels,
        x=xvalues,
        y=yvalues,
        opacity=1,
        transpose=True,
        colorscale=BWR,  # * coloring by the ratio of negtative/positive points
        colorbar={'title': '', 'titleside': 'right', 'len': colorbar_len, 'y':colorbary},
        showscale=showscale)
    l2 = go.Heatmap(
        z=region_labels,
        x=xvalues,
        y=yvalues,
        opacity=0.4,
        transpose=True,
        colorscale=COLOR_Pastel2,  # * random color
        showscale=False,)

    layers.append(l1)
    layers.append(l2)
    if data is not None:
        input_points, true_label = data
        l3 = go.Scatter(
            x=input_points[:, 0], 
            y=input_points[:, 1], 
            mode='markers',
            marker=dict(
                size=3,
                color= torch.squeeze(true_label), 
                colorscale='Viridis')
        )
        layers.append(l3)

    return layers


def compute_distance(data, net, min_distance=None):
    '''
    Compute the distance of a point to its nearest linear region.
    
    Perameters
    ----------
    Input:
      * data: Nx? matrix of N data points
      * net: pytorch network,
    Output:
      * data_out: net(data)
      * min_distance: Nx(Q1+Q2+..+Ql) binary matrix,
    '''
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
        ## TODO: the numerator is z(x) - b in the paper 'Complexity of Linear Regions in Deep Networks'
        net_out = net(data)
        min_d, _ = torch.min(torch.abs(net_out) / torch.norm(net.weight), 1)
        min_distance = torch.min(min_distance, min_d)
        return net_out, min_distance
    else:
        return net(data), min_distance
