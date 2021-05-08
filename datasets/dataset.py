import math
import logging
import torch
import torch.utils.data as Data
import numpy as np
from .synthetic_data.circles import Circles
from .synthetic_data.moons import Moons
from .synthetic_data.sphere import Sphere
from .synthetic_data.spiral import Spiral
# from .iris import Iris
from .eurosat import EuroSat

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
    def __init__(self,
                 name,
                 n_train,
                 n_val,
                 n_test,
                 batch_size=32,
                 num_workers=4,
                 **kwargs) -> None:
        self.name = name
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        if name == 'eurosat':
            self.gen_image_dataset(**kwargs)
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
            assert (self.n_train + self.n_val + self.n_test -
                    1.0) < 1e-5, 'n_train + n_val + n_test must equal to 1!'
            n_train = math.ceil(n_samples * self.n_train)
            n_val = math.ceil(n_samples * self.n_val)
            n_test = n_test = n_samples - n_train - n_val

        self.trainset = get_torch_dataset(
            DATA[self.name].make_data(n_train, **kwargs), self.name)
        self.valset = get_torch_dataset(
            DATA[self.name].make_data(n_val, **kwargs), self.name)
        self.testset = get_torch_dataset(
            DATA[self.name].make_data(n_test, **kwargs), self.name)

    def gen_image_dataset(self, data_dir, **kwargs):
        shuffle = kwargs['shuffle']
        dataset = DATA[self.name](data_dir)
        N = len(dataset.targets)
        if isinstance(self.n_test, int):
            n_test = self.n_test
        elif isinstance(self.n_test, float):
            n_test = int(N * self.n_test)

        # sample labeled
        categorized_idx = [
            list(np.where(np.array(dataset.targets) == i)[0])
            for i in range(dataset.num_classes)
        ]  #[[], [],]

        sample_distrib = np.array(
            [len(idx_group) for idx_group in categorized_idx])
        sample_distrib = sample_distrib / sample_distrib.max()

        if shuffle:
            for i in range(dataset.num_classes):
                np.random.shuffle(categorized_idx[i])

        # rerange indexs following the rule so that labels are ranged like: 0,1,....9,0,....9,...
        # adopted from https://github.com/google-research/fixmatch/blob/79f9fd3e6267035d685864beaec40dd45408ecb0/scripts/create_split.py#L87
        npos = np.zeros(dataset.num_classes, np.int64)
        idx_test = []
        for i in range(n_test):
            c = np.argmax(sample_distrib - npos / max(npos.max(), 1))
            idx_test.append(
                categorized_idx[c][npos[c]])  # the indexs of examples
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
        self.valset = TransformedDataset(dataset, idx_val)
        self.testset = TransformedDataset(dataset, idx_test)

    def gen_dataloader(self):
        kwargs = dict(batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      pin_memory=True,
                      drop_last=False)
        self.train_loader = Data.DataLoader(self.trainset,
                                            shuffle=True,
                                            **kwargs)
        self.val_loader = Data.DataLoader(self.valset, **kwargs)
        self.test_loader = Data.DataLoader(self.testset, **kwargs)


class TransformedDataset(Dataset):
    def __init__(self, dataset, index, transform=None, target_transform=None):
        self.dataset = dataset
        self.data = dataset.data
        self.targets = np.array(dataset.targets)[index]
        self.transform = transform
        self.target_transform = target_transform
        self.index = index

    def __getitem__(self, i):
        img, target = self.dataset[self.index[i]]
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.index)
