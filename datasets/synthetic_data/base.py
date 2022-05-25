import os
from abc import abstractmethod
from typing import Tuple
import torch
import numpy as np
from sklearn import svm
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt


class Base(object):

    @classmethod
    @abstractmethod
    def make_data(cls, *args, **kwargs):
        '''
        Method to generate point data.
        '''
        pass

    @classmethod
    def add_noise(cls, X: 'np.array', noise_ratio: float = 0, noise_level: float = 0, seed: int = 0) -> 'np.array':
        '''   
        Parameter
        ---------
        * X : ndarray of shape (n_samples, 2)
        * noise_ratio: float, betwen 0-1, defines the number of noisy points
        * noise_level: float, betwen 0-1, defines the level a point is diverging from the sample
        * seed: default 0

        Returns
        -------
        * X : ndarray of shape (n_samples, 2)
            The generated samples.
        '''
        if noise_ratio:
            generator = check_random_state(seed)
            noise_n = int(noise_ratio * X.shape[0])
            idx = np.random.choice(np.arange(X.shape[0]), size=noise_n)
            X[idx, :] += generator.normal(scale=noise_level, size=(noise_n, 2))
        return X

    @classmethod
    def sampling_to_plot_LR(cls, X: 'np.array', y: 'np.array', res: int = 200) -> Tuple['torch.Tensor', 'np.array']:
        '''
        Generate a grid of point set for the visualization of linear regions.

        Parameters
        ----------
        * X: the point data, has size (n, 2)
        * y: used to generate the label (label mask) for the grid points by svm.
        * res: defines the number of points in the grid, res * res

        Returns
        ----------
        * torch.Tensor: the grid points, size (200*200, 2)
        * np.array: label of grid points, size (200*200,)
        '''
        # create grid to evaluate model
        minX, minY = X.min(0)
        maxX, maxY = X.max(0)
        xx = np.linspace(minX, maxX, res)
        yy = np.linspace(minY, maxY, res)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        clf = svm.SVC(kernel="rbf", C=100)
        clf.fit(X[:, :2], y)
        prob = clf.decision_function(xy).reshape(XX.shape)
        TH = 1
        decision_boundary = 1 - ((-TH <= prob) & (prob <= TH)).astype(float)
        negtive_class = -2 * (prob < -TH).astype(float)
        grid_labels = decision_boundary + negtive_class
        return torch.tensor(xy).float(), grid_labels

    def plot(self, data: 'np.array', targets: 'np.array', name: str, noise_ratio: float, noise_level: float, save_dir: str='./') -> None:
        '''
        visualize the generated data using matplotlib

        Parameters
        ----------
        * data: the point data, size (n, 2)
        * targets: labels of the point data, (n, )
        * name: name of the data, displayed in the figure title
        * noise_ratio: noise ratio of the data, displayed in the figure title
        * noise_level: noise level of the data, displayed in the figure title
        * save_dir: folder path to save the figure
        '''
        minX, minY = data.min(0)
        maxX, maxY = data.max(0)
        plt.scatter(data[:, 0], data[:, 1], c=targets, cmap=plt.cm.Paired, s=8)
        plt.xlim(minX - 0.1, maxX + 0.1)
        plt.ylim(minY - 0.1, maxY + 0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{name} noise/total points = {noise_ratio} noise(std)= {noise_level}')
        plt.savefig(os.path.join(save_dir, name + '.png'))