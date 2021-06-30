import math
import numpy as np
from sklearn.utils import check_random_state
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


class Spiral:
    NUM_CLASSES = 2

    @classmethod
    def spiral_xy(cls, i, spiral_num, n_samples=1):
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
    
    @classmethod
    def spiral(cls, spiral_num, n_samples=1):
        return [
            Spiral.spiral_xy(i, spiral_num, n_samples)
            for i in range(n_samples * 97)
        ]

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

        x1 = np.array(Spiral.spiral(1, n_samples))
        x2 = np.array(Spiral.spiral(-1, n_samples))
        l1 = np.ones(x1.shape[0])
        l2 = np.zeros(x2.shape[0])
        X = np.concatenate([x1, x2], axis=0)
        y = np.concatenate([l1, l2])[:, None]

        X += generator.normal(scale=width, size=X.shape)
        if noise_ratio:
            X = cls.add_noise(X, noise_ratio, noise_level, seed)
        return X, y

    @classmethod
    def sampling_to_plot_LR(self, X, y, res=200):
        # create grid to evaluate model
        minX, minY = X.min(0)
        maxX, maxY = X.max(0)
        xx = np.linspace(minX, maxX, res)
        yy = np.linspace(minY, maxY, res)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        w1 = res
        w2 = res
        x1, x2 = np.split(X, 2)
        matrix = [[0.0 for i in range(w1)] for j in range(w2)]
        for x, y in x1:
            x = min(int(round(x * w1)), w1 - 1)
            y = min(int(round(y * w2)), w2 - 1)
            matrix[w2 -1 - y][x] = 1
        for x, y in x2:
            x = min(int(round(x * w1)), w1 - 1)
            y = min(int(round(y * w2)), w2 - 1)
            matrix[w2-1 - y][x] = -1

        matrix = np.array(matrix)
        matrix = gaussian_filter(matrix, sigma=5)
        TH = 0.01
        matrix[np.where(np.abs(matrix) < TH)] = 0
        matrix[np.where(matrix > TH)] = 1
        matrix[np.where(matrix < -TH)] = -1

        return xy, np.rot90(matrix, k=-1)
    
    @classmethod
    def make_trajectory(cls, type='same_class', interval=0.001):
        '''
        return the trajectory and its length
        '''
        if type == 'same_class':
            xy = np.array(Spiral.spiral(1, 100))
            x = xy[:, 0]
            y = xy[:, 1]
            distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 )) # euclidean distance of each pair of neighboring points
            traj_len = distance[-1]
            distance = distance/distance[-1]
            fx, fy = interp1d( distance, x ), interp1d( distance, y )

            alpha = np.linspace(0, 1, 5114) # ! 5114 is chosen according to the interval 0.001
            x_regular, y_regular = fx(alpha), fy(alpha)
            return np.concatenate([x_regular[:, None], y_regular[:, None]], axis=1), traj_len
        else:
            x = np.arange(0,1, interval)
            return np.array([x, x]).T, np.sqrt(2*x[-1]**2)