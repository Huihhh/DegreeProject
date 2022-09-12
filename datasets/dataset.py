from datasets.real_data.mnist import Mnist
import math
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from torch.utils.data.sampler import RandomSampler, BatchSampler
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
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

    def __init__(self,
                 name: str,
                 n_train: float,
                 n_val: float,
                 n_test: float,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seed: int = 0,
                 **kwargs) -> None:
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
        * seed: default 0, used for point dataset
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

        logger.info(f"Dataset: {self.name}")

        if name == 'eurosat':
            self.gen_image_dataset(**kwargs)
        elif name == 'mnist':
            self.gen_mnist_dataset(**kwargs)
        else:
            self.gen_point_dataset(**kwargs)

        logger.info(f'>>> #trainset = {len(self.trainset)}')
        logger.info(f'>>> #valset = {len(self.valset)}')
        logger.info(f'>>> #testset = {len(self.testset)}')

    def gen_point_dataset(self, fixed_valset: int = 0, n_samples: int = 1000, **kwargs) -> None:
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
        test_arr = dataset.make_data(n_test, **{**kwargs, 'seed': 20})  #!fixed seed
        # update the exact number of train, val test size
        self.n_train, self.n_val, self.n_test = len(train_arr), len(val_arr), len(test_arr)

        self.grid_data = dataset.sampling_to_plot_LR(*train_arr)
        self.trainset = get_torch_dataset(train_arr, self.name)
        self.valset = get_torch_dataset(val_arr, self.name)
        self.testset = get_torch_dataset(test_arr, self.name)

    def gen_image_dataset(self, data_dir: str, aug_times: int = 1, **kwargs) -> None:
        '''
        Generate image torch dataset

        Parameter
        ---------
        * data_dir: folder for the dataset
        * aug_times: the number factor for augmented data, actual batch size = batch_size * (aug_times +1)
        '''
        eurosat = DATA[self.name](data_dir, self.n_test, self.n_val)
        self.trainset = eurosat.get_dataset('train')
        self.valset = eurosat.get_dataset('val')
        self.testset = eurosat.get_dataset('test')
        # *** stack augmented data ***
        if kwargs.get('use_aug'):
            self.aug_times = aug_times
            logger.info('********* apply data augmentation ***********')
            transform = T.Compose([
                T.RandomHorizontalFlip(p=kwargs['flip_p']),
                T.RandomApply([
                    T.RandomCrop(size=64, padding=int(64 * 0.125), padding_mode='reflect'),
                ], p=0.5),
                T.RandomApply([
                    T.RandomRotation(90),
                ], p=0.5),
                T.RandomAdjustSharpness(0.5),
                T.ToTensor(),
                T.Normalize(mean=eurosat.MEAN, std=eurosat.STD)
            ])
            print('aug_times', aug_times)

            self.trainset_aug = eurosat.get_dataset('train', transform=transform)
            # self.trainset = [self.trainset, trainset_aug]

        # self.trainset = Data.ConcatDataset([self.trainset, dataset_aug])

    def train_dataloader(self) -> Any:
        if self.name == 'eurosat':
            kwargs = dict(num_workers=self.num_workers, pin_memory=False)
            if hasattr(self, 'aug_times'):
                replacement = self.aug_times > 1
                num_samples = self.aug_times * len(self.trainset_aug) if replacement else None
                data_sampler = RandomSampler(self.trainset_aug, replacement=replacement, num_samples=num_samples)
                aug_loader = DataLoader(self.trainset_aug,
                               batch_sampler=BatchSampler(data_sampler, self.batch_size * self.aug_times, drop_last=False),
                               **kwargs)
                train_loader = CombinedLoader([
                    DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, **kwargs),
                    aug_loader
                ])
            else:
                train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, **kwargs)

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
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False,
                          drop_last=False)
        # return CombinedLoader({'test': test_loader, 'val': val_loader})
