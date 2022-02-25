import numpy as np
from sklearn import svm
from sklearn.utils import check_random_state
from .base import Base


class Circles(Base):
    NUM_CLASSES = 2

    @classmethod
    def make_data(cls,
                  n_samples,
                  width,
                  gap,
                  equal_density=False,
                  seed=0,
                  noise_ratio=0,
                  noise_level=0,
                  *args,
                  **kwargs):
        """Make a large circle containing a smaller circle in 2d.
        A simple toy dataset to visualize clustering and classification
        algorithms.
        Read more in the :ref:`User Guide <sample_generators>`.
        Parameters
        ----------
        shuffle : bool, default=True
            Whether to shuffle the samples.
        width : float, default=None
            Standard deviation of Gaussian noise added to the data. width of the circles
        gap : float, default = 0.4
            boundary width between the circles
        seed : int, RandomState instance or None, default=None
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <seed>`.
        factor : float, default=.8
            Scale factor between inner and outer circle in the range `(0, 1)`.
        noise_ratio : float, default=0
            the ratio of extra noise
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            The generated samples.
        y : ndarray of shape (n_samples,)
            The integer labels (0 or 1) for class membership of each sample.
        """
        if gap >= 1 or gap < 0:
            raise ValueError("the width of the gap between circles has to be between 0 and 1.")

        # each class has the same density
        if equal_density:
            ratio = ((1 + width)**2 - (1 - width)**2) / ((gap + width)**2 - (gap - width)**2)
            n_samples_inner = int(1 / (1 + ratio) * n_samples)
        else:
            # each class has the same number of points
            n_samples_inner = n_samples // 2

        n_samples_out = n_samples - n_samples_inner

        # so as not to have the first point = last point, we set endpoint=False
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_inner, endpoint=False)
        outer_circ_x = np.cos(linspace_out)
        outer_circ_y = np.sin(linspace_out)
        inner_circ_x = np.cos(linspace_in) * gap
        inner_circ_y = np.sin(linspace_in) * gap

        X = np.vstack([
            np.append(outer_circ_x, inner_circ_x),
            np.append(outer_circ_y, inner_circ_y),
        ]).T
        y = np.hstack([
            np.zeros(n_samples_out, dtype=np.intp),
            np.ones(n_samples_inner, dtype=np.intp),
        ])
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
            # linspace_out = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
            linspace_out = np.arange(0, 2 * np.pi, interval)
            outer_circ_x = np.cos(linspace_out)
            outer_circ_y = np.sin(linspace_out)
            xy = np.vstack([outer_circ_x, outer_circ_y]).T
            traj_len = np.sqrt( np.ediff1d(xy[:, 0], to_begin=0)**2 + np.ediff1d(xy[:, 1], to_begin=0)**2).sum()
            return xy, traj_len
        else:
            x = np.arange(-1, 1, interval)
            return np.array([x, x]).T, np.sqrt(2 * (x[0] - x[-1])**2)
