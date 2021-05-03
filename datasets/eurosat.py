from PIL import Image
from omegaconf import DictConfig, OmegaConf
import numpy as np

import hydra
import logging
import glob
from collections import Counter
import os
import zipfile
import urllib.request as Request

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms as T
from typing import Any, Callable, Optional, Tuple
import torch.utils.data as Data


TRANSFORM = {
        'mean': (0.5, 0.5, 0.5), #(0.4914, 0.4822, 0.4465),  #
        'std': (0.25, 0.25, 0.25)#}, # (0.2471, 0.2435, 0.2616),  # 
}

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:     
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)       
    else:
        print('This is not zip')

def download_data(_save_path, _url):
    try:
        Request.urlretrieve(_url, _save_path)
        return True
    except:
        print('\nError when retrieving the URL:\n{}'.format(_url))
        return False

logger = logging.getLogger(__name__)

class EuroSat(VisionDataset):
    def __init__(self, CFG, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,) -> None:
        super(EuroSat, self).__init__(CFG.data_dir, transform= transform, target_transform=target_transform)
        self.CFG = CFG
        self.batch_size = CFG.batch_size
        self.num_workers = CFG.num_workers
        self.data = []
        self.targets = []
        class_counter = Counter()

        # load images
        rootdir = hydra.utils.get_original_cwd()
        if not os.path.exists(rootdir + CFG.data_dir):
            download_data(rootdir + '/data/EuroSAT_RGB.zip', 'http://madm.dfki.de/files/sentinel/EuroSAT.zip')
            unzip_file(rootdir + '/data/EuroSAT_RGB.zip', rootdir + '/data/EuroSAT_RGB')
        filepaths = glob.glob(rootdir + CFG.data_dir + '/*/*.jpg')
        for filepath in filepaths:
            class_name = filepath.split('/')[-1].split('_')[0]
            class_counter[class_name] += 1
            img = Image.open(filepath)
            self.data.append([np.array(img)])
            self.targets.extend([class_name])

        self.data = np.concatenate(self.data)
        self.class_to_idx = {_class: i for i, _class in enumerate(class_counter.keys())}
        self.targets = [self.class_to_idx[x] for x in self.targets]
        self.num_classes = len(class_counter.keys())

        self.mean = TRANSFORM['mean']
        self.std = TRANSFORM['std']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean,
                        std=self.std)
        ])

        self.train_test_split(CFG.n_val, CFG.n_test)
        logger.info(f"Dataset: {self.CFG.name}")
        logger.info(f'>>> #trainset = {len(self.trainset)}')
        logger.info(f'>>> #valset = {len(self.valset)}')
        logger.info(f'>>> #testset = {len(self.testset)}')
        self.get_dataloader()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def train_test_split(self, val_size, test_size, shuffle=False):  # TODO: make it a function of the basic class
        N = len(self.targets)
        if isinstance(test_size, int):
            n_test = test_size
        elif isinstance(test_size, float):
            n_test = int(N * test_size)

        # sample labeled
        categorized_idx = [list(np.where(np.array(self.targets) == i)[0]) for i in range(self.num_classes)] #[[], [],]
        
        sample_distrib = np.array([len(idx_group) for idx_group in categorized_idx])
        sample_distrib = sample_distrib/sample_distrib.max()

        if shuffle:
            for i in range(self.num_classes):
                np.random.shuffle(categorized_idx[i])

        # rerange indexs following the rule so that labels are ranged like: 0,1,....9,0,....9,...
        # adopted from https://github.com/google-research/fixmatch/blob/79f9fd3e6267035d685864beaec40dd45408ecb0/scripts/create_split.py#L87
        npos = np.zeros(self.num_classes, np.int64)
        idx_test = []
        for i in range(n_test):
            c = np.argmax(sample_distrib - npos / max(npos.max(), 1))
            idx_test.append(categorized_idx[c][npos[c]]) # the indexs of examples
            npos[c] += 1
        
        idx_train = np.setdiff1d(np.array(np.arange(N)), np.array(idx_test))

        n_val_per_class = int(N * val_size // self.num_classes)
        idx_val = []
        for idxs in categorized_idx:
            idxs = list(set(idxs) - set(idx_test))
            idx = np.random.choice(idxs, n_val_per_class, replace=False)
            idx_val = np.concatenate((idx_val, idx), axis=None)
        idx_val = list(idx_val.astype(int))
        idx_train = np.setdiff1d(idx_train, np.array(idx_val))
        self.trainset = TransformedDataset(self, idx_train)
        self.valset = TransformedDataset(self, idx_val)
        self.testset = TransformedDataset(self, idx_test)

    def get_dataloader(self):
        kwargs = dict(batch_size=self.batch_size,
                      num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.train_loader = Data.DataLoader(
            self.trainset, shuffle=True, **kwargs)
        self.val_loader = Data.DataLoader(self.valset, **kwargs)
        self.test_loader = Data.DataLoader(self.testset, **kwargs)

    def make_points_to_plot_LR(self, *args, **kwargs):
        return self.data, self.targets

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


logger = logging.getLogger(__name__)




if __name__ == '__main__':
    @hydra.main(config_path='../config/dataset/', config_name='eurosat')
    def main(CFG: DictConfig):
        logger.info(OmegaConf.to_yaml(CFG))
        import time
        t1 = time.time()
        CFG = OmegaConf.structured(OmegaConf.to_yaml(CFG))
        CFG.DATASET.seed = 0
        euroset = EuroSat(CFG.DATASET)
        print(time.time() - t1)
    main()
