
from PIL import Image
from hydra.types import TargetConf
from omegaconf import DictConfig, OmegaConf
import numpy as np

import hydra
import logging
import zipfile
import urllib.request as Request
import scipy.io
from sklearn.model_selection import train_test_split

import torch.utils.data as Data
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms as T
from typing import Any, Callable, Optional, Tuple


TRANSFORM = {
        'mean': (0.5, 0.5, 0.5, 0.5), #(0.4914, 0.4822, 0.4465),  #
        'std': (0.25, 0.25, 0.25, 0.25)#}, # (0.2471, 0.2435, 0.2616),  # 
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

class BaseSet(Dataset):
    def __init__(self, data, targets, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,) -> None:
        super(BaseSet, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

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
        
        img = img.reshape(-1)

        return img, target
    

logger = logging.getLogger(__name__)

class Sat(VisionDataset):
    def __init__(self, CFG, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,) -> None:
        super(Sat, self).__init__(CFG.data_dir, transform= transform, target_transform=target_transform)
        self.CFG = CFG
        self.batch_size = CFG.batch_size
        self.num_workers = CFG.num_workers

        # load mat
        train_x, train_y = [], []
        test_x, test_y = [], []
        rootdir = hydra.utils.get_original_cwd()
        for i in CFG.classes:
            mat = scipy.io.loadmat(f'{rootdir}/{CFG.data_dir}/class{i}.mat')
            train_x.append(mat['train_x'])
            train_y.append(np.squeeze(mat['train_y']))
            test_x.append(mat['test_x'])
            test_y.append(np.squeeze(mat['test_y']))
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y)

        # train test split
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=self.CFG.n_val, random_state=self.CFG.seed)
        # get dataset
        self.mean = TRANSFORM['mean']
        self.std = TRANSFORM['std']
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean,
                        std=self.std)
        ])
        
        self.trainset = BaseSet(train_x, train_y, transform=self.transform)
        self.valset = BaseSet(val_x, val_y, transform=self.transform)
        self.testset = BaseSet(test_x, test_y, transform=self.transform)

        logger.info(f"Dataset: {self.CFG.name}")
        self.get_dataloader()

    def get_dataloader(self):
        kwargs = dict(batch_size=self.batch_size,
                      num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.train_loader = Data.DataLoader(
            self.trainset, shuffle=True, **kwargs)
        self.val_loader = Data.DataLoader(self.valset, **kwargs)
        self.test_loader = Data.DataLoader(self.testset, **kwargs)
        
    
    def get_decision_boundary(self):
        # create grid to evaluate model
        h = 0.01
        xx = np.arange(self.minX, self.maxX, h)
        yy = np.arange(self.minY, self.maxY, h)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        temp = self.CFG.noise_ratio
        self.CFG.noise_ratio = None
        X, y = self.DATA[self.CFG.name]()
        self.CFG.noise_ratio = temp
        clf = svm.SVC(kernel='rbf', C=100)
        clf.fit(X[:, :2], y)
        prob = clf.decision_function(xy).reshape(XX.shape)
        TH = 1
        decision_boundary = 1 - \
            ((-TH <= prob) & (prob <= TH)).astype(float)
        negtive_class = -2 * (prob < -TH).astype(float)
        grid_labels = decision_boundary + negtive_class

        if len(self.CFG.increase_dim) > 0:
            xy, grid_labels = self.extend_input([xy, grid_labels])
        return xy, grid_labels


if __name__ == '__main__':
    import sys, os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    print(sys.path)
    @hydra.main(config_path='../config/dataset/', config_name='sat4')
    def main(CFG: DictConfig):
        logger.info(OmegaConf.to_yaml(CFG))
        import time
        t1 = time.time()
        CFG = OmegaConf.structured(OmegaConf.to_yaml(CFG))
        CFG.DATASET.seed = 0
        sat = Sat(CFG.DATASET)
        print(time.time() - t1)
    main()
