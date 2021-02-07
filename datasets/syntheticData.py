import numpy as np
import matplotlib.pyplot as plt
import torch, os
import torch.utils.data as Data


class Dataset(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.total_samples = cfg.total_samples
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.seed = cfg.seed
        self.noise = cfg.noise
        self.r1, self.r2 = cfg.r1, cfg.r2
        self.get_dataloader()


    def gen_circle_data(self):
        # each class has the same density 
        n_noise = int(self.total_samples * self.noise)
        n_samples = self.total_samples - n_noise
        if self.cfg.equal_density:
            ratio = (self.r2[1] - self.r1[0]) / self.r1[1]
            num_class1 = int(1 / (1+ratio) * n_samples)
        else: # even the number of samples per class
            num_class1 = n_samples // 2

        n_list = [num_class1, n_samples - num_class1, n_noise] # number of points per class
        radius_list = [self.r1, self.r2, [self.r1[-1], self.r2[0]]] 
        X, labels = [], []
        for i, (n, rang) in enumerate(zip(n_list, radius_list)):
            theta = np.random.uniform(0, 2*np.pi, n)
            r = np.random.uniform(*rang, n) ** 0.5
            
            x = [r * np.cos(theta)]
            y = [r * np.sin(theta)]
            label = i * np.ones(n) if i != 2 else np.random.randint(0,2, n)

            X.append(np.concatenate([x, y], 0))
            labels.append(label)

        self.data = [np.concatenate(X, 1).transpose(1, 0), np.concatenate(labels)]
        self.minX, self.minY = self.data[0].min(0)
        self.maxX, self.maxY = self.data[0].max(0)
        return self.data

    def plot(self, save_dir='./data'):
        x, l = self.gen_circle_data()
        plt.figure(figsize=(10, 10), dpi=125)
        idxs = np.where(l==1)
        plt.plot(x[:, 0], x[:, 1], 'bo', markersize=1)
        plt.plot(x[idxs, 0], x[idxs, 1], 'ro', markersize=1)
        plt.xlim(self.minX-0.1, self.maxX + 0.1)
        plt.ylim(self.minY-0.1, self.maxY + 0.1)
        _t = np.arange(0, 7, 0.1)
        boundary_colors = ['b-', 'r-']
        boundary_radius = [self.r1[1]**0.5, self.r2[0]**0.5]
        for r, clr in zip(boundary_radius, boundary_colors):
            _x = r * np.cos(_t)
            _y = r * np.sin(_t)
            plt.plot(_x, _y, clr)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Random Scatter')
        # plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(save_dir, self.cfg.name + '.png'))

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
    import os, sys, random
    sys.path.append(os.getcwd())

    @hydra.main(config_path='../config/dataset', config_name='synthetic')
    def main(CFG: DictConfig):
            # # For reproducibility, set random seed
        if CFG.DATASET.seed == 'None':
            CFG.DATASET.seed = random.randint(1, 10000)
        np.random.seed(CFG.DATASET.seed)
        dataset = Dataset(CFG.DATASET)
        dataset.plot()
        np.savetxt('./data/input.txt', dataset.data[0], delimiter=',')
        # x, y = next(iter(dataset.train_loader))
        # print(x)
        # print(y)
    main()
