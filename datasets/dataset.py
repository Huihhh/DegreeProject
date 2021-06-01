import math
import logging
import torch
import torch.utils.data as Data
from torchvision import transforms as T
import numpy as np
from collections import defaultdict
from .synthetic_data.circles import Circles
from .synthetic_data.moons import Moons
from .synthetic_data.sphere import Sphere
from .synthetic_data.spiral import Spiral
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes
# from .iris import Iris
from .eurosat import *

DATA = {
    'circles': Circles,
    'moons': Moons,
    'spiral': Spiral,
    'sphere': Sphere,
    # 'iris': Iris,
    'eurosat': EuroSat
}

logger = logging.getLogger(__name__)


def get_torch_dataset(ndarray, name):
    X = torch.from_numpy(ndarray[0]).float()
    if DATA[name].NUM_CLASSES == 2:
        Y = torch.from_numpy(ndarray[1]).float()
    else:
        Y = torch.from_numpy(ndarray[1]).long()
    return Data.TensorDataset(X, Y)


class Dataset(Data.TensorDataset):
    def __init__(self, name, n_train, n_val, n_test, batch_size=32, num_workers=4, resnet=None, **kwargs) -> None:
        self.name = name
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        if name == 'eurosat':
            self.gen_image_dataset(resnet, **kwargs)
        else:
            self.gen_point_dataset(**kwargs)
        self.gen_dataloader()
        logger.info(f"Dataset: {self.name}")
        logger.info(f'>>> #trainset = {len(self.trainset)}')
        logger.info(f'>>> #valset = {len(self.valset)}')
        logger.info(f'>>> #testset = {len(self.testset)}')

    def gen_point_dataset(self, fixed_valset=0, n_samples=1000, **kwargs):
        if fixed_valset:
            n_train = n_samples
            n_val = n_test = fixed_valset
        else:
            assert (self.n_train + self.n_val + self.n_test - 1.0) < 1e-5, 'n_train + n_val + n_test must equal to 1!'
            n_train = math.ceil(n_samples * self.n_train)
            n_val = math.ceil(n_samples * self.n_val)
            n_test = n_test = n_samples - n_train - n_val

        dataset = DATA[self.name]()
        self.num_classes = 2
        #TODO: sampling_to_plot_LR
        self.sampling_to_plot_LR = dataset.sampling_to_plot_LR
        self.trainset = get_torch_dataset(dataset.make_data(n_train, **kwargs), self.name)
        self.valset = get_torch_dataset(dataset.make_data(n_val, **kwargs), self.name)
        self.testset = get_torch_dataset(dataset.make_data(n_test, **kwargs), self.name)

    def gen_image_dataset(self, resnet, data_dir, **kwargs):
        dataset = DATA[self.name](data_dir)
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
                                                        batch_size=self.batch_size,
                                                        num_workers=self.num_workers,
                                                        pin_memory=True,
                                                        drop_last=False)

    def gen_dataloader(self):
        if self.name == 'eurosat':
            kwargs = dict(num_workers=self.num_workers, pin_memory=True)
            if isinstance(self.trainset, list):
                self.train_loader = [Data.DataLoader(trainset, shuffle=True, **kwargs) for trainset in self.trainset]
            else:
                self.train_loader = [Data.DataLoader(self.trainset, batch_sampler= BatchWeightedRandomSampler(self.trainset, batch_size=self.batch_size), **kwargs)]
            self.val_loader = Data.DataLoader(self.valset, batch_sampler= BatchWeightedRandomSampler(self.valset, batch_size=self.batch_size), **kwargs)
            self.test_loader = Data.DataLoader(self.testset,batch_sampler= BatchWeightedRandomSampler(self.testset, batch_size=self.batch_size), **kwargs)
        else:
            kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False)
            self.train_loader = Data.DataLoader(self.trainset, shuffle=True, **kwargs)
            self.val_loader = Data.DataLoader(self.valset, **kwargs)
            self.test_loader = Data.DataLoader(self.testset, **kwargs)



class TransformedDataset(Dataset):
    def __init__(self, dataset, indexs, transform=None, target_transform=None):
        self.dataset = dataset
        # self.data = dataset.data[indexs]
        # self.targets = np.array(dataset.targets)[indexs]
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
    def __init__(self,  data_source, batch_size, drop_last=True):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.targets = np.array(data_source.dataset.targets)[data_source.indexs]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        nclass = max(self.targets) + 1
        sample_distrib = np.array([len(np.where(self.targets==i)[0]) for i in range(nclass)])
        sample_distrib = sample_distrib/sample_distrib.max()

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
            label.append(class_id[c][npos[c]]) # the indexs of examples
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