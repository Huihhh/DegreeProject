import numbers
import numpy as np
from sklearn.utils import check_random_state
from .base import Base


class Moons(Base):
    NUM_CLASSES = 2

    @classmethod
    def make_data(cls, n_samples, width, gap, noise_ratio=0, noise_level=0, seed=0, *args, **kwargs):
        """ Adapted from sklearn.datasets.make_moons
        Make two interleaving half circles.
        A simple toy dataset to visualize clustering and classification
        algorithms. Read more in the :ref:`User Guide <sample_generators>`.
        Parameters
        ----------
        n_samples : int or tuple of shape (2,), dtype=int, default=100
            If int, the total number of points generated.
            If two-element tuple, number of points in each of two moons.
            .. versionchanged:: 0.23
            Added two-element tuple.
        shuffle : bool, default=True
            Whether to shuffle the samples.
        width : float, default=None
            Standard deviation of Gaussian noise added to the data. width of the circles
        gap : float, default = 0.4
            boundary width between the circles
        seed : int, RandomState instance or None, default=None
            Determines random number generation for dataset shuffling and w.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <seed>`.
        noise_ratio: float, betwen 0-1, defines the number of noisy points
        noise_level: float, betwen 0-1, defines the level a point is diverging from the sample
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            The generated samples.
        y : ndarray of shape (n_samples,)
            The integer labels (0 or 1) for class membership of each sample.
        """
        if isinstance(n_samples, numbers.Integral):
            n_samples_out = n_samples // 2
            n_samples_inner = n_samples - n_samples_out
        else:
            try:
                n_samples_out, n_samples_inner = n_samples
            except ValueError as e:
                raise ValueError('`n_samples` can be either an int or ' 'a two-element tuple.') from e

        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_inner))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_inner)) - gap

        X = np.vstack([np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]).T
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_inner, dtype=np.intp)])
        generator = check_random_state(seed)
        X += generator.normal(scale=width, size=X.shape)
        if noise_ratio:
            X = cls.add_noise(X, noise_ratio, noise_level, seed)
        return X, y[:, None]

    @classmethod
    def make_trajectory(cls, type='same_class', interval=0.001):
        '''
        return the trajectory and its length
        '''
        if type == 'same_class':
            # outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples))
            # outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples))
            outer_circ_x = np.cos(np.arange(0, np.pi, interval))
            outer_circ_y = np.sin(np.arange(0, np.pi, interval))
            xy = np.vstack([outer_circ_x, outer_circ_y]).T
            traj_len = np.sqrt( np.ediff1d(xy[:, 0], to_begin=0)**2 + np.ediff1d(xy[:, 1], to_begin=0)**2).sum()
            return xy, traj_len
        else:
            x = np.arange(-1, 2, 3*interval/np.sqrt(13))
            y = np.arange(-1, 1, 2*interval/np.sqrt(13))[::-1]
            return np.array([x, y]).T, np.sqrt((x[0] - x[-1])**2 + (y[0] - y[-1])**2)
