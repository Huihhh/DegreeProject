'''
check input data:
python datasets/syntheticData.py hydra.run.dir='./outputs/check_datasets'
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import isin
from sklearn.utils.validation import check_random_state
import torch
import os
import logging
import torch.utils.data as Data
from sklearn import datasets, svm
from sklearn.utils import check_random_state, shuffle as util_shuffle
import numbers

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, CFG) -> None:
        super().__init__()
        self.CFG = CFG
        self.total_samples = CFG.total_samples
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
        }
        self.data = self.DATA[self.CFG.name]()
        logger.info('the number of negative points: %d' %
                    len(np.where(self.data[1] == 0)[0]))
        logger.info('the number of positive points: %d' %
                    len(np.where(self.data[1] == 1)[0]))
        self.minX, self.minY = self.data[0].min(0)[:2]
        self.maxX, self.maxY = self.data[0].max(0)[:2]

    def make_circles(self):
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
        self.CFG.factor : float, default=.8
            Scale self.CFG.factor between inner and outer circle in the range `(0, 1)`.
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
        n_noise = int(self.total_samples * self.CFG.noise_ratio)
        n_samples = self.total_samples - n_noise
        equal_density = True
        if equal_density:
            ratio = ((1+w/2)**2 - (1-w/2)**2) / \
                ((boundary_w+w/2)**2 - (boundary_w-w/2)**2)
            n_samples_in = int(1 / (1+ratio) * n_samples)
        else:  # even the number of samples per class
            n_samples_in = n_samples // 2

        n_samples_out = n_samples - n_samples_in

        generator = check_random_state(self.CFG.seed)
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
        if self.CFG.shuffle:
            X, y = util_shuffle(X, y, random_state=generator)

        if w is not None:
            X += generator.normal(scale=w, size=X.shape)

        if self.CFG.noise_ratio:
            X, y = self.add_noise([X, y], generator)

        return X, y

    def make_moons(self):
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

        n_samples = self.total_samples
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

        generator = check_random_state(self.CFG.seed)

        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - \
            np.sin(np.linspace(0, np.pi, n_samples_in)) - self.CFG.boundary_w

        X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                       np.append(outer_circ_y, inner_circ_y)]).T
        y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                       np.ones(n_samples_in, dtype=np.intp)])

        if self.CFG.shuffle:
            X, y = util_shuffle(X, y, random_state=generator)

        if w is not None:
            X += generator.normal(scale=w, size=X.shape)
        if self.CFG.noise_ratio:
            X, y = self.add_noise([X, y], generator)

        if len(self.CFG.increase_dim) > 0:
            X, y = self.extend_input([X, y])
        return X, y

    def add_noise(self, data, generator):
        X, y = data
        noise_n = int(self.CFG.noise_ratio * self.total_samples)
        idx = np.random.choice(np.arange(self.total_samples), size=noise_n)
        X[idx,
            :] += generator.normal(scale=self.CFG.noise_level, size=(noise_n, 2))
        return X, y

    def extend_input(self, data):
        X, y = data
        xx = X.copy()
        for m in self.CFG.increase_dim:
            X = np.concatenate([X, eval(m)(xx)], -1)
        return X, y

    def get_decision_boundary(self):
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
            xy, grid_labels= self.extend_input([xy, grid_labels])
        return xy, grid_labels

    def plot(self, save_dir='./'):
        x, l = self.data
        plt.figure(figsize=(10, 10), dpi=125)
        plt.scatter(x[:, 0], x[:, 1], c=l, cmap=plt.cm.Paired)
        plt.xlim(self.minX-0.1, self.maxX + 0.1)
        plt.ylim(self.minY-0.1, self.maxY + 0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(self.CFG.name)
        plt.savefig(os.path.join(save_dir, self.CFG.name + '.png'))

    def get_dataloader(self):
        X = torch.from_numpy(self.data[0]).float()
        Y = torch.from_numpy(self.data[1]).long()
        dataset = Data.TensorDataset(X, Y)
        trainset, valset, testset = Data.random_split(
            dataset, [self.CFG.n_train, self.CFG.n_val, self.CFG.n_test])
        kwargs = dict(batch_size=self.batch_size,
                      num_workers=self.num_workers, pin_memory=True)
        self.train_loader = Data.DataLoader(trainset, shuffle=True, **kwargs)
        self.val_loader = Data.DataLoader(valset, **kwargs)
        self.test_loader = Data.DataLoader(testset, **kwargs)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    import os
    import sys
    sys.path.append(os.getcwd())

    @hydra.main(config_path='../config/dataset', config_name='synthetic')
    def main(CFG: DictConfig):
        # # For reproducibility, set random seed
        np.random.seed(CFG.DATASET.seed)
        torch.manual_seed(CFG.DATASET.seed)
        torch.cuda.manual_seed_all(CFG.DATASET.seed)
        dataset = Dataset(CFG.DATASET)
        dataset.plot()
        grid_points, grid_labels = dataset.get_decision_boundary()
        # plt.figure()
        # plt.imshow(grid_points.reshape(grid_labels.shape))
        # plt.savefig('./grid_points.png')
        plt.figure()
        plt.imshow(grid_labels)
        plt.savefig('./mask.png')
        np.savetxt('./input.txt', dataset.data[0], delimiter=',')
    main()
