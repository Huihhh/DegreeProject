import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data


class Dataset(object):
    def __init__(self, cfg, r1=(0, 0.75), r2=(1.25, 2)) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_samples = cfg.num_samples
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.seed = cfg.seed
        self.noise = cfg.noise
        self.r1, self.r2 = r1, r2
        self.xrange = [-2, 2]
        self.yrange = [-2, 2]
        self.get_dataloader()


    def gen_circle_data(self):
        np.random.seed(self.seed)

        # each class has the same density 
        if self.cfg.equal_density:
            ratio = (self.r2[1] - self.r1[0]) / self.r1[1]
            num_class1 = int(1 / (1+ratio) * self.num_samples)
        else: # even the number of samples per class
            num_class1 = self.num_samples // 2

        num_class2 = self.num_samples - num_class1
        X, labels = [], []
        for i, (n, rang) in enumerate(zip([num_class1, num_class2], [self.r1, self.r2])):
            theta = np.random.uniform(0, 2*np.pi, n)
            r = np.random.uniform(*rang, n) ** 0.5
            
            x = [r * np.cos(theta)]
            y = [r * np.sin(theta)]
            label = i * np.ones(n)

            X.append(np.concatenate([x, y], 0))
            labels.append(label)

        self.data = [np.concatenate(X, 1).transpose(1, 0), np.concatenate(labels)]
        self.minX, self.minY = self.data[0].min(0)
        self.maxX, self.maxY = self.data[0].max(0)
        return self.data

    def plot(self):
        x, l = self.gen_circle_data()
        plt.figure(figsize=(10, 10), dpi=125)
        idxs = np.where(l==1)
        plt.plot(x[:, 0], x[:, 1], 'bo', markersize=1)
        plt.plot(x[idxs, 0], x[idxs, 1], 'ro', markersize=1)
        plt.xlim(self.minX-0.1, self.maxX + 0.1)
        plt.ylim(self.minY-0.1, self.maxY + 0.1)
        _t = np.arange(0, 7, 0.1)
        for r in [self.r1[1]**0.5, self.r2[0]**0.5]:
            _x = r * np.cos(_t)
            _y = r * np.sin(_t)
            plt.plot(_x, _y, 'g-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Random Scatter')
        # plt.grid(True)
        # plt.show()
        plt.savefig('./data/syntheticData.png')

    def get_dataloader(self):
        X, Y = self.gen_circle_data()
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        dataset = Data.TensorDataset(X, Y)
        trainset, valset, testset = Data.random_split(dataset, [self.cfg.n_train, self.cfg.n_val, self.cfg.n_test], generator=torch.Generator().manual_seed(self.seed))
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.train_loader = Data.DataLoader(trainset, shuffle=False, **kwargs)
        self.val_loader = Data.DataLoader(valset, **kwargs)
        self.test_loader = Data.DataLoader(testset, **kwargs)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    import os, sys
    sys.path.append(os.getcwd())

    @hydra.main(config_path='../config', config_name='config')
    def main(CFG: DictConfig):

        dataset = Dataset(CFG.DATASET)
        dataset.plot()
        np.savetxt('./data/input.txt', dataset.data[0], delimiter=',')
        # x, y = next(iter(dataset.train_loader))
        # print(x)
        # print(y)
    main()
