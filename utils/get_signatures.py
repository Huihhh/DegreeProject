import torch
import torch.nn as nn
import numpy as np
from typing import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import wandb

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
        return net(data), signatures, signatures.shape[1]
    else:
        raise Exception('Unknown operation: ' + str(net))


def plot_linear_regions(trainer):
    grid_data = trainer.grid_points
    trainset = trainer.dataset.trainset
    model = trainer.model
    TH = trainer.CFG.TH
    TH_bounds = trainer.CFG.TH_bounds
    plot_confidence = trainer.CFG.plot_confidence
    xx, yy, grid_labels = grid_data[0][:, 0], grid_data[0][:, 1], grid_data[1]
    img_shape = grid_labels.shape
    net_out, sigs_grid, _ = get_signatures(torch.tensor(grid_data[0]).float().to(device), model)
    net_out = torch.sigmoid(net_out)
    pseudo_label = torch.where(net_out.cpu() > TH, 1.0, 0.0).numpy()
    sigs_grid = np.array([''.join(str(x)
                                  for x in s.tolist()) for s in sigs_grid])

    region_sigs = list(np.unique(sigs_grid))
    total_regions = {}
    total_regions['density'] = len(region_sigs)
    region_ids = np.random.permutation(total_regions['density'])

    sigs_grid_dict = dict(zip(region_sigs, region_ids))
    base_color_labels = np.array(
        [sigs_grid_dict[sig] for sig in sigs_grid])
    base_color_labels = base_color_labels.reshape(grid_labels.shape).T

    grid_labels = grid_labels.reshape(-1)

    train_data, train_labels = trainset.tensors
    _, sigs_train, _ = get_signatures(train_data.to(device), model)
    sigs_train = np.array([''.join(str(x)
                                   for x in s.tolist()) for s in sigs_train])
    sigs_train = Counter(sigs_train)
    boundary_regions, blue_regions, red_regions = defaultdict(
        int), defaultdict(int), defaultdict(int)
    if isinstance(TH_bounds, float):
        bounds = [-TH_bounds, TH_bounds]
    else:
        bounds = TH_bounds
    for i, key in enumerate(sigs_grid_dict):
        idx = np.where(sigs_grid == key)
        region_labels = grid_labels[idx]
        ratio = sum(region_labels) / region_labels.size
        if ratio > bounds[1]:
            red_regions['density'] += 1
            red_regions['area'] += region_labels.size
            red_regions['non_empty_regions'] += int(sigs_train[key] > 0)
        elif ratio < bounds[0]:
            blue_regions['density'] += 1
            blue_regions['area'] += region_labels.size
            blue_regions['non_empty_regions'] += int(sigs_train[key] > 0)
        else:
            boundary_regions['density'] += 1
            boundary_regions['area'] += region_labels.size
            boundary_regions['non_empty_regions'] += int(
                sigs_train[key] > 0)

    red_regions['ratio'] = red_regions['density'] / \
        (red_regions['area'] + 1e-6)
    blue_regions['ratio'] = blue_regions['density'] / \
        (blue_regions['area'] + 1e-6)
    boundary_regions['ratio'] = boundary_regions['density'] / \
        (boundary_regions['area'] + 1e-6)
    total_regions['non_empty_regions'] = boundary_regions['non_empty_regions'] + \
        red_regions['non_empty_regions'] + \
        blue_regions['non_empty_regions']
    total_regions['non_empty_ratio'] = total_regions['non_empty_regions'] / \
        total_regions['density']
    logger.info(f"[Linear regions/area] \n \
                                                #around the boundary: {boundary_regions['density']} \n \
                                                #red region:          {red_regions['density']} \n \
                                                #blue region:         {blue_regions['density'] } \n \
                                                #total regions:       {total_regions['density']} ")

    if plot_LR:
        # save confidence map
        if plot_confidence:
            fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
            ax = ax.flatten()
        else:
            fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
            ax = ax.flatten()
            plt.rcParams['figure.figsize'] = (4.0, 8.0)

        plt.tight_layout(w_pad=-0.2, h_pad=0.8)
        kwargs = dict(
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
        )
        c = 1
        for lables, name in zip([pseudo_label.squeeze(), grid_labels], ['pseudo_label', 'true_label']):
            color_labels = np.zeros(lables.shape)
            for i, key in enumerate(sigs_grid_dict):
                idx = np.where(sigs_grid == key)
                region_labels = lables[idx]
                ratio = sum(region_labels) / region_labels.size
                color_labels[idx] = ratio

            color_labels = color_labels.reshape(img_shape).T

            cmap = mpl.cm.bwr
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
            ax[c].imshow(color_labels, cmap=cmap, norm=norm, alpha=1, **kwargs)
            ax[c].imshow(base_color_labels, cmap=plt.get_cmap('Pastel2'), alpha=0.6, **kwargs)
            ax[c].set_title(name)
            ax[c].set(aspect=1)
            c -= 1

        # linear regions colored by true labels with sample points
        # TODO:change points color
        ax[2].imshow(color_labels, cmap=cmap, norm=norm, alpha=0.8, **kwargs)
        ax[2].imshow(base_color_labels, cmap=plt.get_cmap('Pastel2'), alpha=0.5, **kwargs)
        ax[2].scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=1)
        ax[2].set_title('true label')
        ax[2].set(xlim=[xx.min(), xx.max()], ylim=[yy.min(), yy.max()], aspect=1)
        if plot_confidence:
            confidence = net_out.reshape(img_shape).detach().cpu().numpy()
            ax0 = ax[-1].scatter(xx, yy, c=confidence, vmin=0, vmax=1)
            ax[-1].set(xlim=[xx.min(), xx.max()], ylim=[yy.min(), yy.max()], aspect=1)
            ax[-1].set_title('confidence map')
            fig.colorbar(ax0, ax=ax.ravel().tolist())
        plt.close(fig)

    return total_regions, red_regions, blue_regions, boundary_regions, fig
