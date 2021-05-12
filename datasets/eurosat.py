from PIL import Image
import numpy as np

import hydra
import glob
from collections import Counter
import os
import zipfile
import urllib.request as Request

import torch
import torch.utils.data as Data
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms as T
from typing import Any, Callable, Optional, Tuple

TRANSFORM = {
    'mean': (0.5, 0.5, 0.5),  #(0.4914, 0.4822, 0.4465),  #
    'std': (0.25, 0.25, 0.25)  #}, # (0.2471, 0.2435, 0.2616),  # 
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


class EuroSat(VisionDataset):
    def __init__(
        self,
        data_dir,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(EuroSat, self).__init__(data_dir, transform=transform, target_transform=target_transform)
        self.data = []
        self.targets = []
        class_counter = Counter()

        # load images
        rootdir = hydra.utils.get_original_cwd()
        if not os.path.exists(rootdir + data_dir):
            download_data(rootdir + '/data/EuroSAT_RGB.zip', 'http://madm.dfki.de/files/sentinel/EuroSAT.zip')
            unzip_file(rootdir + '/data/EuroSAT_RGB.zip', rootdir + '/data/EuroSAT_RGB')
        filepaths = glob.glob(rootdir + data_dir + '/*/*.jpg')
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
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])

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

    def sampling_to_plot_LR(self, mean, var, noise_size, *args, **kwargs):
        # idx = np.random.permutation(len(self.data))
        # subset = Data.Subset(self, idx[:noise_size])
        noise = np.random.normal(mean, var**0.5, [noise_size, 3, 64, 64])
        noise_label = np.zeros(len(noise)) * -1 #np.random.randint(0, 9, size=len(noise))  #TODO: how to set the label of noise?
        noise = torch.from_numpy(noise).float()
        noise_label = torch.from_numpy(noise_label).long()
        dataset = Data.TensorDataset(noise, noise_label)
        # concatDataset = Data.ConcatDataset([self, dataset])
        loader = Data.DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, drop_last=False)

        return loader
