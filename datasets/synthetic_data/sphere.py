import numpy as np
from sklearn.utils import check_random_state


class Sphere:
    NUM_CLASSES = 2

    @classmethod
    def make_data(cls,
                  n_samples,
                  radius,
                  width,
                  center0,
                  center1,
                  noise_ratio=0,
                  noise_level=0,
                  seed=0,
                  *args,
                  **kwargs):
        """
        Make two spheres.
        Parameters
        ----------
        radius : number > 0
        width : float, default=None
            Standard deviation of Gaussian noise added to the data. width of the sphere
        center0, center1: center points of the two spheres
        seed : int, RandomState instance or None, default=None
            Determines random number generation for dataset shuffling and w.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <seed>`.
        n_samples: int, default=50
            the number of points: n_samples * n_samples *2
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            The generated samples.
        y : ndarray of shape (n_samples,)
            The integer labels (0 or 1) for class membership of each sample.
        """
        n = complex(imag=n_samples)
        generator = check_random_state(seed)

        def make_sphere(radius, width, sign, center, n=100j):
            phi, theta = np.mgrid[0:np.pi:n, 0:2 * np.pi:n]
            x = radius * np.sin(phi) * np.cos(theta) + center[0]
            y = radius * np.sin(phi) * np.sin(theta) + center[1]
            z = radius * np.cos(phi) + center[2]
            x += generator.normal(scale=width, size=x.shape)
            y += generator.normal(scale=width, size=y.shape)
            z += generator.normal(scale=width, size=z.shape)
            data = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T
            targets = np.ones_like(x.reshape(-1)) * sign
            return data, targets[:, None]

        x1, y1 = make_sphere(radius, width, sign=0, center=center0, n=n)
        x2, y2 = make_sphere(radius, width, sign=1, center=center1, n=n)
        X = np.vstack([x1, x2])
        y = np.vstack([y1, y2])
        if noise_ratio:
            X = cls.add_noise(X, noise_ratio, noise_level, seed)
        return X, y

    @classmethod
    def make_grid_with_labels(self, X, y, h=0.001):
        #TODO: make points to plot linear regions
        return X, y
