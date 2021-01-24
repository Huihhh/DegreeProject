import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data


class Dataset(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_samples = cfg.num_samples
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.seed = cfg.seed
        self.noise = cfg.noise
        self.get_dataloader()


    def gen_circle_data(self, r1=0.75, r2=1.25):
        # each class has the same density 
        if self.cfg.equal_density:
            ratio = (r2**2 - r1**2) / r1**2
            num_class1 = int(1 / (1+ratio) * self.num_samples)
        else: # even the number of samples per class
            num_class1 = self.num_samples // 2

        num_class2 = self.num_samples - num_class1
        X, labels = [], []
        for i, n in enumerate([num_class1, num_class2]):
            t1 = np.random.random(n) # random numbers between [0, 1] that control the phase
            t2 = np.random.random(n) # random numbers between [0, 1] that control the amplitude
            if i == 0:
                r = r1 * t2
                l = np.ones(num_class1)
            else: 
                r = r2 + t2
                l = np.zeros(num_class2)
            x = [r * np.cos(2 * np.pi * t1) + self.noise * ( -r1 + 2*r1 * np.random.random(n))]
            y = [r * np.sin(2 * np.pi * t1) + self.noise * ( -r1 + 2*r1 * np.random.random(n))]
            X.append(np.concatenate([x, y], 0))
            labels.append(l)

        self.data = [np.concatenate(X, 1).transpose(1, 0), np.concatenate(labels)]
        self.minX, self.minY = self.data[0].min(0)
        self.maxX, self.maxY = self.data[0].max(0)
        return self.data

    def plot(self):
        r1 = 0.75
        r2 = 1.25
        x, l = self.gen_circle_data(r1=r1, r2=r2)
        plt.figure(figsize=(10, 10), dpi=125)
        idxs = np.where(l==1)
        plt.plot(x[:, 0], x[:, 1], 'ro', markersize=1)
        plt.plot(x[idxs, 0], x[idxs, 1], 'bo', markersize=1)
        _t = np.arange(0, 7, 0.1)
        for r in [r1, r2]:
            _x = r * np.cos(_t)
            _y = r * np.sin(_t)
            plt.plot(_x, _y, 'g-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Random Scatter')
        # plt.grid(True)
        # plt.show()
        plt.savefig('./data.png')

    def get_dataloader(self):
        X, Y = self.gen_circle_data()
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        dataset = Data.TensorDataset(X, Y)
        trainset, valset, testset = Data.random_split(dataset, [self.cfg.n_train, self.cfg.n_val, self.cfg.n_test], generator=torch.Generator().manual_seed(self.seed))
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.train_loader = Data.DataLoader(trainset, shuffle=True, **kwargs)
        self.val_loader = Data.DataLoader(valset, **kwargs)
        self.test_loader = Data.DataLoader(testset, **kwargs)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os, sys
    sys.path.append(os.getcwd())

    @hydra.main(config_path='../config', config_name='config')
    def main(CFG: DictConfig):

        dataset = Dataset(CFG.DATASET)
        dataset.plot()
        x, y = next(iter(dataset.train_loader))
        print(x)
        print(y)
    main()
