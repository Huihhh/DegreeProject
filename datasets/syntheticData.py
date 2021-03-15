'''
check input data:
python datasets/syntheticData.py hydra.run.dir='./outputs/check_datasets' +DATASET.seed=0
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
import torch.utils.data as Data
from sklearn import datasets, svm

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
            'circles_fill': self.make_circles_fill,
            'circles': self.make_circles,
            'moons': self.make_moons,
        }
        self.data = self.DATA[self.CFG.name](**self.CFG)
        logger.info('the number of negative points: %d' %
                    len(np.where(self.data[1] == 0)[0]))
        logger.info('the number of positive points: %d' %
                    len(np.where(self.data[1] == 1)[0]))
        self.minX, self.minY = self.data[0].min(0)
        self.maxX, self.maxY = self.data[0].max(0)

    def make_circles_fill(self, noise=0.05, r1=(0, 0.75), r2=(1.25, 3), equal_density=False, **kwargs):
        # each class has the same density
        n_noise = int(self.total_samples * noise)
        n_samples = self.total_samples - n_noise
        if equal_density:
            ratio = (r2[1] - r2[0]) / r1[1]
            num_class1 = int(1 / (1+ratio) * n_samples)
        else:  # even the number of samples per class
            num_class1 = n_samples // 2

        # number of points per class
        n_list = [num_class1, n_samples - num_class1, n_noise]
        radius_list = [r1, r2, [r1[-1], r2[0]]]
        X, labels = [], []
        for i, (n, rang) in enumerate(zip(n_list, radius_list)):
            theta = np.random.uniform(0, 2*np.pi, n)
            r = np.random.uniform(*rang, n) ** 0.5

            x = [r * np.cos(theta)]
            y = [r * np.sin(theta)]
            label = i * np.ones(n) if i != 2 else np.random.randint(0, 2, n)

            X.append(np.concatenate([x, y], 0))
            labels.append(label)

        labels = np.concatenate(labels)
        return np.concatenate(X, 1).transpose(1, 0), labels

    def make_circles(self, factor=1, noise=0.05, **kwargs):
        return datasets.make_circles(n_samples=self.total_samples, factor=factor, noise=noise)

    def make_moons(self, noise=0.05, **kwargs):
        return datasets.make_moons(n_samples=self.total_samples, noise=noise)

    def get_decision_boundary(self):
        # create grid to evaluate model
        h = 0.01
        xx = np.arange(self.minX, self.maxX, h)
        yy = np.arange(self.minY, self.maxY, h)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        if self.CFG.name == 'circles_fill':
            def compare(point):
                x, y = point
                if x**2 + y**2 <= self.CFG.r1[-1]:
                    return -1
                elif x**2 + y**2 >= self.CFG.r2[0]:
                    return 1
                else:
                    return 0
            grid_labels = np.array(list(map(compare, xy))).reshape(XX.shape)
        else:
            X, y = self.data
            clf = svm.SVC(kernel='rbf', C=100)
            clf.fit(X, y)
            prob = clf.decision_function(xy).reshape(XX.shape)
            TH = 1
            decision_boundary = 1 - ((-TH <= prob) & (prob <= TH)).astype(float)
            negtive_class = -2 * (prob < -TH).astype(float)
            grid_labels = decision_boundary + negtive_class
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
        trainset, valset, testset = Data.random_split(dataset, [self.CFG.n_train, self.CFG.n_val, self.CFG.n_test])
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.train_loader = Data.DataLoader(trainset, shuffle=False, **kwargs)
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
        if 'seed' in CFG.DATASET:
            seed = CFG.DATASET.seed
        else:
            seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset = Dataset(CFG.DATASET)
        dataset.plot()
        grid_points, grid_labels = dataset.get_decision_boundary()
        plt.figure()
        plt.imshow(grid_points.reshape(grid_labels.shape))
        plt.savefig('./grid_points.png')
        plt.figure()
        plt.imshow(grid_labels)
        plt.savefig('./mask.png')
        np.savetxt('./input.txt', dataset.data[0], delimiter=',')
    main()
