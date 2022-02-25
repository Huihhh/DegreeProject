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
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from typing import Any, Callable, Optional, Tuple

TRANSFORM = {
    'mean': (0.3444, 0.3803, 0.4078),  #(0.4914, 0.4822, 0.4465),  #
    'std': (0.2037, 0.1366, 0.1148)  #}, # (0.2471, 0.2435, 0.2616),  # 
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

        # # load images
        rootdir = hydra.utils.get_original_cwd()
        if not os.path.exists(rootdir + data_dir):
            download_data(rootdir + '/data/EuroSAT_RGB.zip', 'http://madm.dfki.de/files/sentinel/EuroSAT.zip')
            unzip_file(rootdir + '/data/EuroSAT_RGB.zip', rootdir + '/data/EuroSAT_RGB')
        filepaths = sorted(glob.glob(rootdir + data_dir + '/*/*.jpg')) ##!!for reproducibility, sort them!
        for filepath in filepaths:
            class_name = filepath.split('/')[-1].split('_')[0]
            class_counter[class_name] += 1
            img = Image.open(filepath)
            self.data.append([np.array(img)])
            self.targets.extend([class_name])

        self.data = np.concatenate(self.data)
        self.class_to_idx = {_class: i for i, _class in enumerate(class_counter.keys())}
        self.classes = list(self.class_to_idx.keys())
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

    def sampling_to_plot_LR(self, noise_size, **kwargs):
        # idx = np.random.permutation(len(self.data))
        # subset = Data.Subset(self, idx[:noise_size])
        noise = []
        for i in range(3):
            noise.append(np.random.normal(TRANSFORM['mean'][i], TRANSFORM['std'][i], [noise_size, i, 64, 64]))
        noise = np.concatenate(noise, axis=1)

        noise_label = np.zeros(len(noise)) * -1  #np.random.randint(0, 9, size=len(noise))  #TODO: how to set the label of noise?
        noise = torch.from_numpy(noise).float()
        noise_label = torch.from_numpy(noise_label).long()
        dataset = Data.TensorDataset(noise, noise_label)
        loader = Data.DataLoader(dataset, **kwargs)
        return loader


if __name__ == '__main__':
    from omegaconf import DictConfig, OmegaConf
    import os
    import sys
    sys.path.append(os.getcwd())
    import matplotlib.pyplot as plt
    def restore_stats(img): 
        mean = TRANSFORM['mean']
        mean = torch.tensor(mean).unsqueeze(dim=1).unsqueeze(dim=1)
        std = TRANSFORM['std']
        std = torch.tensor(std).unsqueeze(dim=1).unsqueeze(dim=1)
        img = img * std + mean
        return T.ToPILImage()(img).convert('RGB')

    def array_to_image(arr):
        arr = np.array(arr.transpose(0, 2))
        if arr.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        arr = np.clip(arr, low_clip, 1.0)
        arr = np.uint8(arr * 255)
        return Image.fromarray(arr)

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG), '\n')
        dataset = EuroSat(data_dir = '/data/EuroSAT_RGB/2750')
        dataloader = iter(dataset.sampling_to_plot_LR(2))
        noise = dataloader.next()[0].squeeze()
        img = dataset[1][0]
        img_noise = noise + img
        noise = restore_stats(noise)
        img_noise = restore_stats(img_noise)
        img = restore_stats(img)
        plt.subplot(131)
        plt.gca().set_title('raw image')
        plt.imshow(img)
        plt.subplot(132)
        plt.gca().set_title('image + Gaussian noise')
        plt.imshow(img_noise)        
        plt.subplot(133)
        plt.gca().set_title('Gaussian noise')
        plt.imshow(noise)
        print(os.getcwd())
        plt.savefig('../img_with_noise1.png')
        print('ok')

    main()
