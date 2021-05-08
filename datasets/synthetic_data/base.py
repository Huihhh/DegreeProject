import os
import numpy as np
from sklearn import svm
from sklearn.utils import check_random_state
from abc import abstractmethod
import matplotlib.pyplot as plt

class Base(object):
    @classmethod
    @abstractmethod
    def make_data(cls, *args, **kwargs):
        pass

    @classmethod
    def add_noise(cls, X, noise_ratio=0, noise_level=0, seed=0):
        """        
        noise_ratio: float, betwen 0-1, defines the number of noisy points
        noise_level: float, betwen 0-1, defines the level a point is diverging from the sample
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            The generated samples.
        """
        if noise_ratio:
            generator = check_random_state(seed)
            noise_n = int(noise_ratio * X.shape[0])
            idx = np.random.choice(np.arange(X.shape[0]), size=noise_n)
            X[idx, :] += generator.normal(scale=noise_level, size=(noise_n, 2))
        return X

    @classmethod
    def sampling_to_plot_LR(cls, X, y, h=0.01):
        # create grid to evaluate model
        minX, minY = X.min(0)
        maxX, maxY = X.max(0)
        xx = np.arange(minX, maxX, h)
        yy = np.arange(minY, maxY, h)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        clf = svm.SVC(kernel="rbf", C=100)
        clf.fit(X[:, :2], y)
        prob = clf.decision_function(xy).reshape(XX.shape)
        TH = 1
        decision_boundary = 1 - ((-TH <= prob) & (prob <= TH)).astype(float)
        negtive_class = -2 * (prob < -TH).astype(float)
        grid_labels = decision_boundary + negtive_class
        return xy, grid_labels

    def plot(self, data, targets, name, noise_ratio, noise_level, save_dir='./'):
        minX, minY = data.min(0)
        maxX, maxY = data.max(0)
        plt.scatter(data[:, 0], data[:, 1], c=targets, cmap=plt.cm.Paired, s=8)
        plt.xlim(minX-0.1, maxX + 0.1)
        plt.ylim(minY-0.1, maxY + 0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(
            f'{name} noise/total points = {noise_ratio} noise(std)= {noise_level}')
        plt.savefig(os.path.join(save_dir, name + '.png'))