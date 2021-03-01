import numpy as np
import matplotlib.pyplot as plt
import torch, os
import torch.utils.data as Data
from sklearn import datasets, svm


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
        self.DATA = {
            'circles_fill': self.make_circles_fill,
            'circles': self.make_circles,
            'moons': self.make_moons,
        }
        self.data = self.DATA[cfg.name]()
        self.minX, self.minY = self.data[0].min(0)
        self.maxX, self.maxY = self.data[0].max(0)
        self.get_dataloader()


    def make_circles_fill(self):
        # each class has the same density 
        n_noise = int(self.total_samples * self.noise)
        n_samples = self.total_samples - n_noise
        if self.cfg.equal_density:
            ratio = (self.r2[1] - self.r2[0]) / self.r1[1]
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

        labels = np.concatenate(labels)
        print('the number of negative points: ', len(np.where(labels==0)[0]))
        print('the number of positive points: ', len(np.where(labels==1)[0]))
        return np.concatenate(X, 1).transpose(1, 0), labels

    def make_circles(self):
        return datasets.make_circles(n_samples=self.total_samples, factor=self.cfg.factor, noise=self.noise)



    def get_decision_boundary(self):
        # create grid to evaluate model
        h = 0.01
        xx = np.arange(self.minX, self.maxX, h)
        yy = np.arange(self.minY, self.maxY, h)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        if self.cfg.name == 'circles_fill':            
            def compare(point):
                x, y = point
                if x**2 + y**2 <=self.cfg.r1[-1]:
                    return -1
                elif x**2 + y**2 >=self.cfg.r2[0] and x**2 + y**2 <=self.cfg.r2[-1]:
                    return 1
                else: 
                    return 0  
            grid_labels = np.array(list(map(compare, xy))).reshape(XX.shape)
        else:            
            X, y = self.data
            clf = svm.SVC(kernel='rbf', C=100)
            clf.fit(X, y)
            prob = clf.decision_function(xy).reshape(XX.shape)
            # prob = np.rot90(prob, k=-3)
            TH = 1
            decision_boundary = 1 - ((-TH<=prob) & (prob<=TH)).astype(float)
            negtive_class = -2 * (prob<-TH).astype(float)
            grid_labels = decision_boundary + negtive_class
        return xy, grid_labels

    def make_moons(self):
        return datasets.make_moons(n_samples=self.total_samples, noise=self.noise)


    def plot(self, save_dir='./'):
        x, l = self.data
        plt.figure(figsize=(10, 10), dpi=125)
        plt.scatter(x[:, 0], x[:, 1], c=l, cmap=plt.cm.Paired)
        plt.xlim(self.minX-0.1, self.maxX + 0.1)
        plt.ylim(self.minY-0.1, self.maxY + 0.1)
        # _t = np.arange(0, 7, 0.1)
        # boundary_colors = ['b-', 'r-']
        # boundary_radius = [self.r1[1]**0.5, self.r2[0]**0.5]
        # for r, clr in zip(boundary_radius, boundary_colors):
        #     _x = r * np.cos(_t)
        #     _y = r * np.sin(_t)
        #     plt.plot(_x, _y, clr)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Random Scatter')
        # plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(save_dir, self.cfg.name + '.png'))

    def get_dataloader(self):
        X = torch.from_numpy(self.data[0]).float()
        Y = torch.from_numpy(self.data[1]).long()
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
        grid_points, grid_labels = dataset.get_decison_boundary()
        print(grid_labels)
        plt.imshow(grid_labels)
        plt.colorbar()
        plt.savefig('./mask.png')
        # dataset.plot()
        np.savetxt('./input.txt', dataset.data[0], delimiter=',')
    main()
