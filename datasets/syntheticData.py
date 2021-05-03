'''
check input data:
python datasets/syntheticData.py hydra.run.dir='./outputs/check_datasets'
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import math
import logging
import torch.utils.data as Data
from sklearn import svm
from sklearn.utils import check_random_state, shuffle as util_shuffle
from scipy.ndimage import gaussian_filter
import numbers

logger = logging.getLogger(__name__)


class SyntheticData(object):
    def __init__(self, CFG) -> None:
        super().__init__()
        self.CFG = CFG
        self.seed = CFG.seed
        self.n_train = CFG.n_train
        self.n_val = CFG.n_val
        self.n_test = CFG.n_test
        self.batch_size = CFG.batch_size
        self.num_workers = CFG.num_workers
        self.get_dataset()
        self.get_dataloader()

    def get_dataset(self):
        logger.info(f'***** Preparing data: {self.CFG.name} *****')
        self.DATA = {
            'circles': self.make_circles,
            'moons': self.make_moons,
            'spiral': self.make_spiral,
            'sphere': self.make_sphere
        }

        def get_torch_dataset(np_data):
            X = torch.from_numpy(np_data[0]).float()
            Y = torch.from_numpy(np_data[1]).float()
            return Data.TensorDataset(X, Y)

        if self.CFG.name in ['spiral', 'sphere']:
            if self.CFG.fixed_valset:
                Nfactor_train = self.CFG.Nfactor
                Nfactor_val = Nfactor_test = self.CFG.fixed_val_factor
            else:
                assert (self.n_train + self.n_val + self.n_test -
                        1.0) < 1e-5, 'n_train + n_val + n_test must equal to 1!'
                Nfactor_train = math.ceil(self.CFG.Nfactor * self.n_train)
                Nfactor_val = math.ceil(self.CFG.Nfactor * self.n_val)
                Nfactor_test = math.ceil(self.CFG.Nfactor * self.n_test)
            self.trainset = get_torch_dataset(
                self.DATA[self.CFG.name](**{**self.CFG, 'Nfactor': Nfactor_train}))
            self.valset = get_torch_dataset(
                self.DATA[self.CFG.name](**{**self.CFG, 'Nfactor': Nfactor_val, 'seed': self.CFG.seed+1}))
            self.testset = get_torch_dataset(
                self.DATA[self.CFG.name](**{**self.CFG, 'Nfactor': Nfactor_test, 'seed': 20}))

        else:
            assert (self.n_train + self.n_val + self.n_test -
                    1.0) < 1e-5, 'n_train + n_val + n_test must equal to 1!'
            n_train = int(self.CFG.total_samples * self.n_train)
            n_val = int(self.CFG.total_samples * self.n_val)
            n_test = self.CFG.total_samples - n_train - n_val
            self.trainset = get_torch_dataset(self.DATA[self.CFG.name](
                n_samples=n_train, seed=self.CFG.seed, shuffle=True))
            self.valset = get_torch_dataset(self.DATA[self.CFG.name](
                n_samples=n_val, seed=self.CFG.seed+1, shuffle=False))
            self.testset = get_torch_dataset(
                self.DATA[self.CFG.name](n_samples=n_test, seed=20, **self.CFG))

        logger.info('the number of negative training points: %d' %
                    len(np.where(self.trainset.tensors[1].numpy() == 0)[0]))
        logger.info('the number of positive training points: %d' %
                    len(np.where(self.trainset.tensors[1].numpy() == 1)[0]))
        self.minX, self.minY = self.trainset.tensors[0].min(0)[0][:2]
        self.maxX, self.maxY = self.trainset.tensors[0].max(0)[0][:2]

    def make_circles(self, n_samples=2000, seed=0, shuffle=True, *args, **kwargs):
        """Make a large circle containing a smaller circle in 2d.
        A simple toy dataset to visualize clustering and classification
        algorithms.
        Read more in the :ref:`User Guide <sample_generators>`.
        Parameters
        ----------
        shuffle : bool, default=True
            Whether to shuffle the samples.
        w : float, default=None
            Standard deviation of Gaussian noise added to the data.
        seed : int, RandomState instance or None, default=None
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <seed>`.
        Nfactor : float, default=.8
            Scale Nfactor between inner and outer circle in the range `(0, 1)`.
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            The generated samples.
        y : ndarray of shape (n_samples,)
            The integer labels (0 or 1) for class membership of each sample.
        """
        w = self.CFG.width
        boundary_w = self.CFG.boundary_w
        if self.CFG.boundary_w >= 1 or self.CFG.boundary_w < 0:
            raise ValueError("boundary width has to be between 0 and 1.")

        # each class has the same density
        # n_noise = int(self.CFG.total_samples * self.CFG.noise_ratio)
        # n_samples = self.CFG.total_samples
        if self.CFG.equal_density:
            ratio = ((1+w/2)**2 - (1-w/2)**2) / \
                ((boundary_w+w/2)**2 - (boundary_w-w/2)**2)
            n_samples_in = int(1 / (1+ratio) * n_samples)
        else:  # even the number of samples per class
            n_samples_in = n_samples // 2

        n_samples_out = n_samples - n_samples_in

        generator = check_random_state(seed)
        # so as not to have the first point = last point, we set endpoint=False
        linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        outer_circ_x = np.cos(linspace_out)
        outer_circ_y = np.sin(linspace_out)
        inner_circ_x = np.cos(linspace_in) * self.CFG.boundary_w
        inner_circ_y = np.sin(linspace_in) * self.CFG.boundary_w

        X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                       np.append(outer_circ_y, inner_circ_y)]).T
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                       np.ones(n_samples_in, dtype=np.intp)])
        if shuffle:
            X, y = util_shuffle(X, y, random_state=generator)

        if w is not None:
            X += generator.normal(scale=w, size=X.shape)

        if self.CFG.noise_ratio:
            X, y = self.add_noise([X, y], generator)

        return X, y[:, None]

    def make_moons(self, n_samples=2000, seed=0, shuffle=True,  *args, **kwargs):
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
        w : float, default=None
            Standard deviation of Gaussian w added to the data.
        seed : int, RandomState instance or None, default=None
            Determines random number generation for dataset shuffling and w.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <seed>`.
        noise_ratio: number betwen 0-1, defines the number of noisy points
        noise_level: number betwen 0-1, defines the level a point is diverging from the sample
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            The generated samples.
        y : ndarray of shape (n_samples,)
            The integer labels (0 or 1) for class membership of each sample.
        """

        # n_samples = self.CFG.total_samples
        w = self.CFG.width
        if isinstance(n_samples, numbers.Integral):
            n_samples_out = n_samples // 2
            n_samples_in = n_samples - n_samples_out
        else:
            try:
                n_samples_out, n_samples_in = n_samples
            except ValueError as e:
                raise ValueError('`n_samples` can be either an int or '
                                 'a two-element tuple.') from e

        generator = check_random_state(seed)

        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - \
            np.sin(np.linspace(0, np.pi, n_samples_in)) - self.CFG.boundary_w

        X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                       np.append(outer_circ_y, inner_circ_y)]).T
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                       np.ones(n_samples_in, dtype=np.intp)])

        if shuffle:
            X, y = util_shuffle(X, y, random_state=generator)

        if w is not None:
            X += generator.normal(scale=w, size=X.shape)
        if self.CFG.noise_ratio:
            X, y = self.add_noise([X, y], generator)

        if len(self.CFG.increase_dim) > 0:
            X, y = self.extend_input([X, y])
        return X, y[:, None]

    def make_spiral(self, Nfactor=50, seed=0, shuffle=True, *args, **kwargs):
        def spiral_xy(i, spiral_num, Nfactor=1):
            """
            Create the data for a spiral.

            Arguments:
                i runs from 0 to 96
                spiral_num is 1 or -1
            """
            φ = i/(16*Nfactor) * math.pi
            r = 6.5 * ((104*Nfactor - i)/(104*Nfactor))
            x = (r * math.cos(φ) * spiral_num)/13 + 0.5
            y = (r * math.sin(φ) * spiral_num)/13 + 0.5
            return (x, y)

        def spiral(spiral_num, Nfactor=1):
            return [spiral_xy(i, spiral_num, Nfactor) for i in range(Nfactor * 97)]

        self.x1 = np.array(spiral(1, Nfactor))
        self.x2 = np.array(spiral(-1, Nfactor))
        l1 = np.ones(self.x1.shape[0])
        l2 = np.zeros(self.x2.shape[0])
        X = np.concatenate([self.x1, self.x2], axis=0)
        y = np.concatenate([l1, l2])

        generator = check_random_state(seed)
        if shuffle:
            X, y = util_shuffle(X, y, random_state=generator)

        if self.CFG.width is not None:
            X += generator.normal(scale=self.CFG.width, size=X.shape)

        if self.CFG.noise_ratio:
            X, y = self.add_noise([X, y], generator)
        return X, y[:, None]

    def make_sphere(self, Nfactor=50, seed=0, *args, **kwargs):
        generator = check_random_state(seed)
        n = complex(imag=Nfactor)

        def make_sphere(r, w, sign, center, n=100j):
            phi, theta = np.mgrid[0:np.pi:n, 0:2*np.pi:n]
            x = r * np.sin(phi) * np.cos(theta) + center[0]
            y = r * np.sin(phi) * np.sin(theta) + center[1]
            z = r * np.cos(phi) + center[2]
            x += generator.normal(scale=w, size=x.shape)
            y += generator.normal(scale=w, size=y.shape)
            z += generator.normal(scale=w, size=z.shape)
            data = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T
            targets = np.ones_like(x.reshape(-1)) * sign
            return data, targets[:, None]

        x1, y1 = make_sphere(self.CFG.r, self.CFG.w, sign=0, center=self.CFG.center0, n=n)
        x2, y2 = make_sphere(self.CFG.r, self.CFG.w, sign=1, center=self.CFG.center1, n=n)
        data = np.vstack([x1, x2])
        targets = np.vstack([y1, y2])
        return data, targets

    def add_noise(self, data, generator):
        X, y = data
        noise_n = int(self.CFG.noise_ratio * X.shape[0])
        idx = np.random.choice(np.arange(X.shape[0]), size=noise_n)
        X[idx,
            :] += generator.normal(scale=self.CFG.noise_level, size=(noise_n, 2))
        return X, y

    def extend_input(self, data):
        X, y = data
        xx = X.copy()
        for m in self.CFG.increase_dim:
            X = np.concatenate([X, eval(m)(xx)], -1)
        return X, y

    def make_points_to_plot_LR(self, use_grid=True):
        if use_grid:
            return self.make_grid_points_with_labels()
        else:
            if self.CFG.name in ['spiral', 'sphere']:
                return self.DATA[self.CFG.name](Nfactor=self.CFG.Nfactor*2)
            else:
                return self.DATA[self.CFG.name](n_samples=self.CFG.total_samples*2)

    def make_grid_points_with_labels(self):
        # create grid to evaluate model
        h = 0.01
        xx = np.arange(self.minX, self.maxX, h)
        yy = np.arange(self.minY, self.maxY, h)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        temp = self.CFG.noise_ratio
        self.CFG.noise_ratio = None
        X, y = self.DATA[self.CFG.name]()
        self.CFG.noise_ratio = temp
        clf = svm.SVC(kernel='rbf', C=100)
        clf.fit(X[:, :2], y)
        prob = clf.decision_function(xy).reshape(XX.shape)
        TH = 1
        decision_boundary = 1 - \
            ((-TH <= prob) & (prob <= TH)).astype(float)
        negtive_class = -2 * (prob < -TH).astype(float)
        grid_labels = decision_boundary + negtive_class

        if len(self.CFG.increase_dim) > 0:
            xy, grid_labels = self.extend_input([xy, grid_labels])

        return xy, grid_labels

    def make_grid_points_with_labels_spiral(self):
        # create grid to evaluate model
        h = 0.01
        xx = np.arange(self.minX, self.maxX, h)
        yy = np.arange(self.minY, self.maxY, h)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        w1 = xx.shape[0]
        w2 = yy.shape[0]
        matrix = [[0.0 for i in range(w1)]
                  for j in range(w2)]
        for x, y in self.x1:
            x = min(int(round(x * w1)), w1 - 1)
            y = min(int(round(y * w2)), w2 - 1)
            matrix[1 - y][x] = 1
        for x, y in self.x2:
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

    def plot(self, save_dir='./'):
        x, l = self.trainset.tensors
        plt.scatter(x[:, 0], x[:, 1], c=l, cmap=plt.cm.Paired, s=8)
        plt.xlim(self.minX-0.1, self.maxX + 0.1)
        plt.ylim(self.minY-0.1, self.maxY + 0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(
            f'{self.CFG.name} noise/total points:{self.CFG.noise_ratio} noise(std):{self.CFG.noise_level}')
        plt.savefig(os.path.join(save_dir, self.CFG.name + '.png'))

    def get_dataloader(self):
        kwargs = dict(batch_size=self.batch_size,
                      num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.train_loader = Data.DataLoader(
            self.trainset, shuffle=True, **kwargs)
        self.val_loader = Data.DataLoader(self.valset, **kwargs)
        self.test_loader = Data.DataLoader(self.testset, **kwargs)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    import os
    import sys
    sys.path.append(os.getcwd())

    @hydra.main(config_path='../config', config_name='config')
    def main(CFG: DictConfig):
        # # For reproducibility, set random seed
        np.random.seed(CFG.DATASET.seed)
        torch.manual_seed(CFG.DATASET.seed)
        torch.cuda.manual_seed_all(CFG.DATASET.seed)
        dataset = SyntheticData(CFG.DATASET)
        dataset.get_dataloader()

        dataset.plot()
        grid_points, grid_labels = dataset.make_grid_points_with_labels()
        # plt.figure()
        # plt.imshow(np.rot90(grid_labels))
        plt.savefig('./mask.png')
        # np.savetxt('./input.txt', dataset.data[0], delimiter=',')
    main()
