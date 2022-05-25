from datasets.real_data.mnist import Mnist
import math
import logging
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from torch.utils.data.sampler import Sampler
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import numpy as np
from collections import defaultdict
from .synthetic_data.circles import Circles
from .synthetic_data.moons import Moons
from .synthetic_data.spiral import Spiral

# from .iris import Iris
from datasets.real_data.eurosat import *

DATA = {
    'circles': Circles,
    'moons': Moons,
    'spiral': Spiral,
    # 'iris': Iris,
    'eurosat': EuroSat,
    'mnist': Mnist
}

logger = logging.getLogger(__name__)


def get_torch_dataset(ndarray, name):
    X = torch.from_numpy(ndarray[0]).float()
    if DATA[name].NUM_CLASSES == 2:
        Y = torch.from_numpy(ndarray[1]).float()
    else:
        Y = torch.from_numpy(ndarray[1]).long()
    return TensorDataset(X, Y)


class Dataset(pl.LightningDataModule):
    def __init__(self, name: str, n_train: float, n_val: float, n_test: float, batch_size: int=32, num_workers: int=4, seed: int=0, **kwargs) -> None:
        '''
        Lightning Data Module

        Parameter
        ----------
        * name
        * n_train: float, 0-1.0
        * n_val: float, 0-1.0
        * n_test: float, 0-1.0
        * batch_size: default 32
        * num_workers: default 4
        * seed: default 0
        '''
        super().__init__()
        self.name = name
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.kwargs = kwargs
        if name == 'eurosat':
            self.gen_image_dataset(**kwargs)
        elif name == 'mnist':
            self.gen_mnist_dataset(**kwargs)
        else:
            self.gen_point_dataset(**kwargs)

        logger.info(f"Dataset: {self.name}")
        logger.info(f'>>> #trainset = {len(self.trainset)}')
        logger.info(f'>>> #valset = {len(self.valset)}')
        logger.info(f'>>> #testset = {len(self.testset)}')

    def gen_point_dataset(self, fixed_valset: int=0, n_samples: int=1000, **kwargs) -> None:
        '''
        Generate point torch dataset

        Parameter
        ----------
        * fixed_valset: optional, the size of valset and testset
        * n_samples: the number of samples for Circles and Moons, the number for Spiral will be 2*97*n_samples
        '''
        if fixed_valset:
            n_train = n_samples
            n_val = n_test = fixed_valset
        else:
            assert (self.n_train + self.n_val + self.n_test - 1.0) < 1e-5, 'n_train + n_val + n_test must equal to 1!'
            n_train = math.ceil(n_samples * self.n_train)
            n_val = math.ceil(n_samples * self.n_val)
            n_test = n_samples - n_train - n_val

        dataset = DATA[self.name]()
        self.num_classes = 2
        #TODO: sampling_to_plot_LR
        self.make_data = dataset.make_data
        train_arr = dataset.make_data(n_train, seed=self.seed, **kwargs)
        val_arr = dataset.make_data(n_val, **{**kwargs, 'seed': 21})
        test_arr = dataset.make_data(n_test, **{**kwargs, 'seed': 20})#!fixed seed 
        # update the exact number of train, val test size
        self.n_train, self.n_val, self.n_test = len(train_arr), len(val_arr), len(test_arr)

        self.grid_data = dataset.sampling_to_plot_LR(*train_arr)
        self.trainset = get_torch_dataset(train_arr, self.name)
        self.valset = get_torch_dataset(val_arr, self.name)
        self.testset = get_torch_dataset(test_arr, self.name)

    def gen_image_dataset(self, data_dir: str, **kwargs) -> None:
        '''
        Generate image torch dataset

        Parameter
        ---------
        * data_dir: folder for the dataset
        '''
        TRANSFORM = {
            'mean': (0.3444, 0.3803, 0.4078),  #(0.4914, 0.4822, 0.4465),  #
            'std': (0.2037, 0.1366, 0.1148)  #}, # (0.2471, 0.2435, 0.2616),  # 
        }
        dataset = DATA[self.name](data_dir)
        self.classes = dataset
        self.num_classes = dataset.num_classes
        N = len(dataset.targets)
        if isinstance(self.n_test, int):
            n_test = self.n_test
        elif isinstance(self.n_test, float):
            n_test = int(N * self.n_test)

        # sample labeled
        categorized_idx = [list(np.where(np.array(dataset.targets) == i)[0])
                           for i in range(dataset.num_classes)]  #[[], [],]

        sample_distrib = np.array([len(idx_group) for idx_group in categorized_idx])
        sample_distrib = sample_distrib / sample_distrib.max()

        if kwargs['shuffle']:
            for i in range(dataset.num_classes):
                np.random.shuffle(categorized_idx[i])

        # rerange indexs following the rule so that labels are ranged like: 0,1,....9,0,....9,...
        # adopted from https://github.com/google-research/fixmatch/blob/79f9fd3e6267035d685864beaec40dd45408ecb0/scripts/create_split.py#L87
        npos = np.zeros(dataset.num_classes, np.int64)
        idx_test = []
        for i in range(n_test):
            c = np.argmax(sample_distrib - npos / max(npos.max(), 1))
            idx_test.append(categorized_idx[c][npos[c]])  # the indexs of examples
            npos[c] += 1

        idx_train = np.setdiff1d(np.array(np.arange(N)), np.array(idx_test))

        n_val_per_class = int(N * self.n_val // dataset.num_classes)
        idx_val = []
        for idxs in categorized_idx:
            idxs = list(set(idxs) - set(idx_test))
            idx = np.random.choice(idxs, n_val_per_class, replace=False)
            idx_val = np.concatenate((idx_val, idx), axis=None)
        idx_val = list(idx_val.astype(int))
        idx_train = np.setdiff1d(idx_train, np.array(idx_val))

        self.trainset = TransformedDataset(dataset, idx_train)
        # *** stack augmented data ***
        if 'use_aug' in kwargs.keys() and kwargs['use_aug']:
            logger.info('********* apply data augmentation ***********')
            mean = TRANSFORM['mean']
            std = TRANSFORM['std']
            transform = T.Compose([
                T.RandomHorizontalFlip(p=kwargs['flip_p']),
                T.RandomCrop(size=64, padding=int(64 * 0.125), padding_mode='reflect'),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
            trainset_aug = TransformedDataset(dataset, idx_train, transform=transform)
            self.trainset = [self.trainset, trainset_aug]

        # self.trainset = Data.ConcatDataset([self.trainset, dataset_aug])
        self.valset = TransformedDataset(dataset, idx_val)
        self.testset = TransformedDataset(dataset, idx_test)

        self.noise_loader = dataset.sampling_to_plot_LR(noise_size=len(idx_train),
                                                        # h=64, w=64, stats=TRANSFORM,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.num_workers,
                                                        pin_memory=False,
                                                        drop_last=False)

    def gen_mnist_dataset(self, data_dir: str, **kwargs) -> None:
        TRANSFORM = {
            'mean': (0.1307, 0.1307, 0.1307),  #(0.4914, 0.4822, 0.4465),  #
            'std': (0.3081, 0.3081, 0.3081)  #}, # (0.2471, 0.2435, 0.2616),  # 
        }
        dataset = DATA[self.name](data_dir)
        self.classes = dataset
        self.num_classes = dataset.num_classes
        self.testset = dataset.testset
        N = len(dataset)
        if isinstance(self.n_val, int):
            n_val = self.n_val
        elif isinstance(self.n_val, float):
            n_val = int(N * self.n_val)

        # sample labeled
        categorized_idx = [list(np.where(np.array(dataset.trainset.targets) == i)[0])
                           for i in range(dataset.num_classes)]  #[[], [],]

        sample_distrib = np.array([len(idx_group) for idx_group in categorized_idx])
        sample_distrib = sample_distrib / sample_distrib.max()

        if kwargs['shuffle']:
            for i in range(dataset.num_classes):
                np.random.shuffle(categorized_idx[i])

        # rerange indexs following the rule so that labels are ranged like: 0,1,....9,0,....9,...
        # adopted from https://github.com/google-research/fixmatch/blob/79f9fd3e6267035d685864beaec40dd45408ecb0/scripts/create_split.py#L87
        npos = np.zeros(dataset.num_classes, np.int64)
        idx_val = []
        for i in range(n_val):
            c = np.argmax(sample_distrib - npos / max(npos.max(), 1))
            idx_val.append(categorized_idx[c][npos[c]])  # the indexs of examples
            npos[c] += 1

        idx_train = np.setdiff1d(np.array(np.arange(N)), np.array(idx_val))

        self.trainset = TransformedDataset(dataset.trainset, idx_train)
        # *** stack augmented data ***
        if 'use_aug' in kwargs.keys() and kwargs['use_aug']:
            logger.info('********* apply data augmentation ***********')
            mean = TRANSFORM['mean'][0]
            std = TRANSFORM['std'][0]
            transform = T.Compose([
                T.RandomHorizontalFlip(p=kwargs['flip_p']),
                T.RandomCrop(size=28, padding=int(28 * 0.125), padding_mode='reflect'),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
            trainset_aug = TransformedDataset(dataset.trainset, idx_train, transform=transform)
            self.trainset = [self.trainset, trainset_aug]

        # self.trainset = Data.ConcatDataset([self.trainset, dataset_aug])
        self.valset = TransformedDataset(dataset.trainset, idx_val)

        self.noise_loader = self.sampling_to_plot_LR(noise_size=len(idx_train),
                                                        h=28, w=28, stats=TRANSFORM,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.num_workers,
                                                        pin_memory=False,
                                                        drop_last=False)

    def sampling_to_plot_LR(self, noise_size, h, w, stats, channel_size=1, **kwargs):
        # idx = np.random.permutation(len(self.data))
        # subset = Data.Subset(self, idx[:noise_size])
        noise = []
        for i in range(channel_size):
            noise.append(np.random.normal(stats['mean'][i], stats['std'][i], [noise_size, i, h, w]))
        noise = np.concatenate(noise, axis=1)
        noise_label = np.zeros(len(noise)) * -1  #np.random.randint(0, 9, size=len(noise))  #TODO: how to set the label of noise?
        noise = torch.from_numpy(noise).float()
        noise_label = torch.from_numpy(noise_label).long()
        dataset = TensorDataset(noise, noise_label)
        loader = DataLoader(dataset, **kwargs)
        return loader


    def train_dataloader(self) -> Any:
        if self.name == 'eurosat':
            kwargs = dict(num_workers=self.num_workers, pin_memory=False)
            if isinstance(self.trainset, list):
                train_loader = [DataLoader(trainset, shuffle=True, **kwargs) for trainset in self.trainset]
            else:
                train_loader = DataLoader(self.trainset,
                                          batch_sampler=BatchWeightedRandomSampler(self.trainset,
                                                                                   batch_size=self.batch_size),
                                          **kwargs)

        else:
            kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, drop_last=False)
            train_loader = DataLoader(self.trainset, shuffle=True, **kwargs)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          drop_last=False)

    def test_dataloader(self):
        if self.name == 'eurosat':
            kwargs = dict(num_workers=self.num_workers, pin_memory=False)
            test_loader = DataLoader(self.testset,
                                     batch_sampler=BatchWeightedRandomSampler(self.testset, batch_size=self.batch_size),
                                     **kwargs)
            # val_loader = DataLoader(self.trainset,
            #                         batch_sampler=BatchWeightedRandomSampler(self.trainset, batch_size=self.batch_size),**kwargs)
        else:
            test_loader = DataLoader(self.testset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=False, #?would it cost more mem?
                                     drop_last=False)
            # val_loader = DataLoader(self.trainset,
            #                         batch_size=self.batch_size,
            #                         num_workers=self.num_workers,
            #                         pin_memory=False,
            #                         drop_last=False)
        # return CombinedLoader({'test': test_loader, 'val': val_loader})
        return test_loader


class TransformedDataset(Dataset):
    def __init__(self, dataset, indexs, transform=None, target_transform=None):
        self.dataset = dataset
        self.data = dataset.data[indexs]
        self.targets = np.array(dataset.targets)[indexs]
        self.transform = transform
        self.target_transform = target_transform
        self.indexs = indexs

    def __getitem__(self, i):
        img, target = self.dataset[self.indexs[i]]
        # to return a PIL Image

        if self.transform is not None:
            img, target = self.dataset.data[self.indexs[i]], self.dataset.targets[self.indexs[i]]
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indexs)


class BatchWeightedRandomSampler(Sampler):
    '''Samples elements for a batch with given probabilites of each element'''
    def __init__(self, data_source, batch_size, drop_last=True):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got " "drop_last={}".format(drop_last))
        self.targets = np.array(data_source.targets)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        nclass = max(self.targets) + 1
        sample_distrib = np.array([len(np.where(self.targets == i)[0]) for i in range(nclass)])
        sample_distrib = sample_distrib / sample_distrib.max()

        class_id = defaultdict(list)
        for idx, c in enumerate(self.targets):
            class_id[c].append(idx)

        assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
        class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]

        for i in range(nclass):
            np.random.shuffle(class_id[i])

        # rerange indexs following the rule so that labels are ranged like: 0,1,....9,0,....9,...
        # adopted from https://github.com/google-research/fixmatch/blob/79f9fd3e6267035d685864beaec40dd45408ecb0/scripts/create_split.py#L87
        npos = np.zeros(nclass, np.int64)
        label = []
        for i in range(len(self.targets)):
            c = np.argmax(sample_distrib - npos / max(npos.max(), 1))
            label.append(class_id[c][npos[c]])  # the indexs of examples
            npos[c] += 1

        batch = []
        for idx in label:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.targets) // self.batch_size
        else:
            return (len(self.targets) + self.batch_size - 1) // self.batch_size