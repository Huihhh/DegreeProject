import torch
import torch.nn as nn

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)

class LinearRegions(object):
    def __init__(self, CFG, ) -> None:
        super().__init__()
        self.CFG = CFG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def get_signatures(self, data, net):
        '''
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
        '''
        if type(net) is nn.Sequential or issubclass(type(net), nn.Sequential):
            # Sequential: go over each layer and record activation patterns
            signatures = []
            sizes = []
            for op in net:
                data, op_sig, sz = self.get_signatures(data, op)
                signatures.append(op_sig)
                sizes.append(sz)
            return data, torch.cat(signatures, dim=1), sizes
        elif (type(net) is nn.Linear) or (type(net) is nn.BatchNorm1d):
            # Linear: no non-linearities occur
            return net(data), torch.zeros(data.shape[0], 0, device=self.device, dtype=torch.int8), 0
        elif type(net) is nn.ReLU or type(net) is nn.LeakyReLU:
            # ReLu: each neuron creates a non-linearity
            signatures = (data > 0).type(torch.int8)
            return net(data), signatures, signatures.shape[1]
        else:
            raise Exception('Unknown operation: ' + str(net))

    def get_grid_sigs(self, data, net):
        if isinstance(data, np.array):
            data = torch.tensor(data).float().to(self.device)
        xx, yy = data[:, 0], data[:, 1]
        net_out, grid_sigs, _ = self.get_signatures(data, net)
        net_out = torch.sigmoid(net_out)
        pseudo_label = torch.where(net_out.cpu() > self.CFG.TH, torch.tensor(1), torch.tensor(-1)).numpy()
        grid_sigs = np.array([''.join(str(x) for x in s.tolist()) for s in grid_sigs])
        unique_regions = list(np.unique(grid_sigs))
        n_regions = len(unique_regions)
        region_ids = np.random.permutation(n_regions)
        
        region_idx_dict = dict(zip(unique_regions, region_ids))                
        base_color_labels = np.array([region_idx_dict[sig] for sig in grid_sigs])
        base_color_labels = base_color_labels.reshape(data.shape).T

        grid_labels = data.reshape(-1)
          

    def compute_density(self, region_idx_dict, grid_sigs, grid_labels):
        boundary_regions, blue_regions, red_regions = defaultdict(int), defaultdict(int), defaultdict(int)
        for i, key in enumerate(region_idx_dict):
                idx = np.where(grid_sigs == key)
                region_labels = grid_labels[idx]
                ratio = sum(region_labels) / region_labels.size
                if ratio > self.CFG.TH_bounds[1]:
                    red_regions['count'] += 1
                    red_regions['area'] += region_labels.size
                elif ratio < self.CFG.TH_bounds[0]:
                    blue_regions['count'] += 1
                    blue_regions['area'] += region_labels.size
                else:
                    boundary_regions['count'] += 1
                    boundary_regions['area'] += region_labels.size
        return red_regions, blue_regions, boundary_regions

    def plot(self, data, labels, region_idx_dict, grid_sigs):
        xx, yy = data[:, 0], data[:, 1]
        kwargs = dict(
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
        )
        color_labels = np.zeros(labels.shape)
        for i, key in enumerate(region_idx_dict):
            idx = np.where(grid_sigs == key)
            region_labels = labels[idx]
            ratio = sum(region_labels) / region_labels.size
            color_labels[idx] = ratio

        color_labels = color_labels.reshape(data.shape).T

        plt.figure()
        cmap = mpl.cm.bwr
        norm = mpl.colors.BoundaryNorm(self.CFG.TH_bounds, cmap.N, extend='both')
        plt.imshow(color_labels, cmap=cmap, norm=norm, alpha=1, **kwargs)
        plt.imshow(base_color_labels, cmap=plt.get_cmap('Pastel2'), alpha=0.6, **kwargs)
        if self.CFG.plot_points:
            input_points, labels = self.dataset.data
            plt.scatter(input_points[:, 0], input_points[:, 1], c=labels, linewidths=0.5)

        plt.savefig(self.save_folder / f'{name}_epoch{epoch}.png')

    def plot_signatures(self, data, epoch, name='Linear_regions'):  
        xx, yy = data[:, 0], data[:, 1]

        for lables, name in zip([grid_labels, pseudo_label.squeeze()], ['true_label', 'pseudo_label']):
            color_labels = np.zeros(lables.shape)
            for i, key in enumerate(region_idx_dict):
                idx = np.where(grid_sigs == key)
                region_labels = lables[idx]
                ratio = sum(region_labels) / region_labels.size
                color_labels[idx] = ratio

            color_labels = color_labels.reshape(data.shape).T

            plt.figure()
            cmap = mpl.cm.bwr
            norm = mpl.colors.BoundaryNorm(self.CFG.TH_bounds, cmap.N, extend='both')
            plt.imshow(color_labels, cmap=cmap, norm=norm, alpha=1, **kwargs)
            plt.imshow(base_color_labels, cmap=plt.get_cmap('Pastel2'), alpha=0.6, **kwargs)
            if self.CFG.plot_points:
                input_points, labels = self.dataset.data
                plt.scatter(input_points[:, 0], input_points[:, 1], c=labels, linewidths=0.5)

            plt.savefig(self.save_folder / f'{name}_epoch{epoch}.png')


        # save confidence map
        if self.CFG.plot_confidence:
            confidence = net_out.reshape(data.shape).detach().cpu().numpy()
            plt.scatter(xx, yy, c=confidence, vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(self.save_folder / f'confidenc_epoch{epoch}.png')