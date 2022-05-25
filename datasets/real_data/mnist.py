import os
from typing import Callable, Optional
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import MNIST
from torchvision import transforms as T
import torch
import hydra


TRANSFORM = {
    'mean': (0.1307, 0.1307, 0.1307),  #(0.4914, 0.4822, 0.4465),  #
    'std': (0.3081, 0.3081, 0.3081)  #}, # (0.2471, 0.2435, 0.2616),  # 
}


class Mnist(VisionDataset):
    def __init__(self, data_dir: str, transform: Optional[Callable]=None) -> None:
        '''
        Minist dataset

        Parameters
        -----------
        * data_dir: data folder directory
        * transform: optional
        '''
        rootdir = hydra.utils.get_original_cwd()
        data_dir = os.path.join(rootdir, data_dir)
        downloadFlag = not os.path.exists(data_dir + '/MNIST/raw')

        self.mean = TRANSFORM['mean']
        self.std = TRANSFORM['std']
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean[0], std=self.std[0]), lambda x: torch.reshape(x, (-1,))])
        self.trainset = MNIST(data_dir, train=True, transform=self.transform, download=downloadFlag)
        self.testset = MNIST(data_dir, train=False, transform=self.transform, download=False)

        self.num_classes = max(self.testset.targets) + 1
        self.class_to_idx = self.trainset.class_to_idx


    def __len__(self) -> int:
        return len(self.trainset)