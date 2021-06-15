import math
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import numpy as np
import wandb
from collections import defaultdict
from .synthetic_data.circles import Circles
from .synthetic_data.moons import Moons
from .synthetic_data.sphere import Sphere
from .synthetic_data.spiral import Spiral
from utils.upload_data_artifacts import create_artifacts

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
    return TensorDataset(X, Y)


class Dataset(pl.LightningDataModule):
    def __init__(self, name, n_train, n_val, n_test, batch_size=32, num_workers=4, resnet=None, **kwargs) -> None:
        super().__init__()
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
        self.make_data = dataset.make_data
        self.grid_data = dataset.sampling_to_plot_LR(*dataset.make_data(n_train, **kwargs))
        self.trainset = get_torch_dataset(dataset.make_data(n_train, **kwargs), self.name)
        self.valset = get_torch_dataset(dataset.make_data(n_val, **kwargs), self.name)
        self.testset = get_torch_dataset(dataset.make_data(n_test, **kwargs), self.name)

    def gen_image_dataset(self, resnet, data_dir, **kwargs):
        TRANSFORM = {
            'mean': (0.3444, 0.3803, 0.4078),  #(0.4914, 0.4822, 0.4465),  #
            'std': (0.2037, 0.1366, 0.1148)  #}, # (0.2471, 0.2435, 0.2616),  # 
        }
        run = wandb.init(project=self.name, job_type="train")
        try:
            data = run.use_artifact('split-0.7-0.1-0.2_27000:latest', type="balanced_data")
        except:
            rootdir = hydra.utils.get_original_cwd()
            create_artifacts(rootdir, data_dir)
            data = run.use_artifact('split-0.7-0.1-0.2_27000:latest', type="balanced_data")

        data_dir = data.download()
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        test_dir = os.path.join(data_dir, "test")
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=TRANSFORM['mean'], std=TRANSFORM['std'])])
        self.trainset = EuroSat(train_dir, transform=transform)
        self.valset = EuroSat(val_dir, transform=transform)
        self.testset = EuroSat(test_dir, transform=transform)

        # # *** stack augmented data ***
        # if 'use_aug' in kwargs.keys() and kwargs['use_aug']:
        #     logger.info('********* apply data augmentation ***********')
        #     mean = TRANSFORM['mean']
        #     std = TRANSFORM['std']
        #     transform = T.Compose([
        #         T.RandomHorizontalFlip(p=kwargs['flip_p']),
        #         T.RandomCrop(size=64, padding=int(64 * 0.125), padding_mode='reflect'),
        #         T.ToTensor(),
        #         T.Normalize(mean=mean, std=std)
        #     ])
        #     trainset_aug = TransformedDataset(dataset, idx_train, transform=transform)
        #     self.trainset = [self.trainset, trainset_aug]

        noise = []
        noise_size = len(self.trainset)
        for i in range(3):
            noise.append(np.random.normal(TRANSFORM['mean'][i], TRANSFORM['std'][i], [noise_size, i, 64, 64]))
        noise = np.concatenate(noise, axis=1)

        noise_label = np.zeros(
            len(noise)) * -1  #np.random.randint(0, 9, size=len(noise))  #TODO: how to set the label of noise?
        noise = torch.from_numpy(noise).float()
        noise_label = torch.from_numpy(noise_label).long()
        dataset = Data.TensorDataset(noise, noise_label)
        self.noise_loader = Data.DataLoader(dataset,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            pin_memory=True)

    def train_dataloader(self) -> Any:
        # if self.name == 'eurosat':
        #     kwargs = dict(num_workers=self.num_workers, pin_memory=True)
        #     if isinstance(self.trainset, list):
        #         train_loader = [DataLoader(trainset, shuffle=True, **kwargs) for trainset in self.trainset]
        #     else:
        #         train_loader = [
        #             DataLoader(self.trainset,
        #                        batch_sampler=BatchWeightedRandomSampler(self.trainset, batch_size=self.batch_size),
        #                        **kwargs)
        #         ]
        # else:
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False)
        train_loader = DataLoader(self.trainset, shuffle=True, **kwargs)
        return [train_loader]

    def val_dataloader(self):
        # if self.name == 'eurosat':
        #     kwargs = dict(num_workers=self.num_workers, pin_memory=True)
        #     return DataLoader(self.valset,
        #                       batch_sampler=BatchWeightedRandomSampler(self.valset, batch_size=self.batch_size),
        #                       **kwargs)
        # else:
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=False)

    def test_dataloader(self):
        # if self.name == 'eurosat':
        #     kwargs = dict(num_workers=self.num_workers, pin_memory=True)
        #     test_loader = DataLoader(self.testset,
        #                              batch_sampler=BatchWeightedRandomSampler(self.testset, batch_size=self.batch_size),
        #                              **kwargs)
        #     val_loader = DataLoader(self.trainset,
        #                             batch_sampler=BatchWeightedRandomSampler(self.trainset, batch_size=self.batch_size),
        #                             **kwargs)
        # else:
        test_loader = DataLoader(self.testset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    drop_last=False)
        val_loader = DataLoader(self.trainset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=False)
        return CombinedLoader({'test': test_loader, 'val': val_loader})


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
    def __init__(self, data_source, batch_size, drop_last=True):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
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