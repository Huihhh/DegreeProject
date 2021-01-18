import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data


class Dataset(object):
    def __init__(self, n_train, n_val, n_test, batch_size=32, num_workers=4, seed=0) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.get_dataloader(n_train, n_val, n_test)


    def gen_circle_data(self, num_samples=1000, r1=0.75, r2=1.25, noise=0.7):
        X, labels = [], []
        for i in range(2):
            t1 = np.random.random(num_samples)
            t2 = np.random.random(num_samples)
            if i == 0:
                r = r1 * t2
                l = np.zeros(num_samples)
            else: 
                r = r2 + t2
                l = np.ones(num_samples)
            x = [r * np.cos(2 * np.pi * t1) + noise * ( -r1 + 2*r1 * np.random.random(num_samples))]
            y = [r * np.sin(2 * np.pi * t1) + noise * ( -r1 + 2*r1 * np.random.random(num_samples))]
            X.append(np.concatenate([x, y], 0))
            labels.append(l)

        self.data = [np.concatenate(X, 1).transpose(1, 0), np.concatenate(labels)]

    def plot(self):
        r1 = 0.75
        r2 = 1.25
        x, l = self.gen_circle_data(r1=r1, r2=r2)
        plt.figure(figsize=(10, 10), dpi=125)
        idxs = np.where(l==1)
        plt.plot(x[:, 0], x[:, 1], 'ro')
        plt.plot(x[idxs, 0], x[idxs, 1], 'bo')
        _t = np.arange(0, 7, 0.1)
        _x = r2 * np.cos(_t)
        _y = r2 * np.sin(_t)
        plt.plot(_x, _y, 'g-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Random Scatter')
        plt.grid(True)
        plt.show()

    def get_dataloader(self, n_train, n_val, n_test):
        self.gen_circle_data()
        X, Y = self.data
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        dataset = Data.TensorDataset(X, Y)
        trainset, valset, testset = Data.random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(self.seed))
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.train_loader = Data.DataLoader(trainset, shuffle=True, **kwargs)
        self.val_loader = Data.DataLoader(valset, **kwargs)
        self.test_loader = Data.DataLoader(testset, **kwargs)


if __name__ == '__main__':
    dataset = Dataset(1400, 400, 200)
    # dataset.plot()
    x, y = next(iter(dataset.train_loader))
    print(x)
    print(y)
