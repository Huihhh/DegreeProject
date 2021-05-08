import math
import numpy as np
from sklearn.utils import check_random_state
from scipy.ndimage import gaussian_filter


class Spiral:
    NUM_CLASSES = 2

    @classmethod
    def make_data(cls, n_samples, width, noise_ratio, noise_level=0, seed=0, *args, **kwargs):
        """
        adopted from: https://conx.readthedocs.io/en/latest/Two-Spirals.html
        n_samples : int, default=50
                controls the number of points: 97 * 2 * n_samples
        width : float, default=None
            Standard deviation of Gaussian noise added to the data. width of the circles
        noise_ratio: float, betwen 0-1, defines the number of noisy points
        noise_level: float, betwen 0-1, defines the level a point is diverging from the sample
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            The generated samples.
        y : ndarray of shape (n_samples,)
            The integer labels (0 or 1) for class membership of each sample.
        """
        generator = check_random_state(seed)

        def spiral_xy(i, spiral_num, n_samples=1):
            """
            Create the data for a spiral.
            Arguments:
                i runs from 0 to 96
                spiral_num is 1 or -1
            """
            φ = i / (16 * n_samples) * math.pi
            r = 6.5 * ((104 * n_samples - i) / (104 * n_samples))
            x = (r * math.cos(φ) * spiral_num) / 13 + 0.5
            y = (r * math.sin(φ) * spiral_num) / 13 + 0.5
            return (x, y)

        def spiral(spiral_num, n_samples=1):
            return [
                spiral_xy(i, spiral_num, n_samples)
                for i in range(n_samples * 97)
            ]

        x1 = np.array(spiral(1, n_samples))
        x2 = np.array(spiral(-1, n_samples))
        l1 = np.ones(x1.shape[0])
        l2 = np.zeros(x2.shape[0])
        X = np.concatenate([x1, x2], axis=0)
        y = np.concatenate([l1, l2])[:, None]

        X += generator.normal(scale=width, size=X.shape)
        if noise_ratio:
            X = cls.add_noise(X, noise_ratio, noise_level, seed)
        return X, y

    @classmethod
    def sampling_to_plot_LR(self, X, y, h=0.01):
        # create grid to evaluate model
        minX, minY = X.min(0)
        maxX, maxY = X.max(0)
        xx = np.arange(minX, maxX, h)
        yy = np.arange(minY, maxY, h)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        w1 = xx.shape[0]
        w2 = yy.shape[0]
        x1, x2 = np.split(X, 2)
        matrix = [[0.0 for i in range(w1)] for j in range(w2)]
        for x, y in x1:
            x = min(int(round(x * w1)), w1 - 1)
            y = min(int(round(y * w2)), w2 - 1)
            matrix[1 - y][x] = 1
        for x, y in x2:
            x = min(int(round(x * w1)), w1 - 1)
            y = min(int(round(y * w2)), w2 - 1)
            matrix[1 - y][x] = -1

        matrix = np.array(matrix)
        matrix = gaussian_filter(matrix, sigma=1)
        TH = 0.01
        matrix[np.where(np.abs(matrix) < TH)] = 0
        matrix[np.where(matrix > TH)] = 1
        matrix[np.where(matrix < -TH)] = -1

        return xy, np.rot90(matrix, k=-1)
