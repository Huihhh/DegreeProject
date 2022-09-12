from itertools import accumulate
import os
import logging
import zipfile
import random
from collections import Counter
from typing import Any, Callable, Iterable, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import hydra
import urllib.request as Request
import torch
import torch.utils.data as Data
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)

def unzip_file(zip_src: str, dst_dir: str) -> None:
    '''
    Unzip .zip file to specific folder

    Parameter
    ---------
    * zip_src: path of the zip file
    * dst_dir: destination directory
    '''
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in tqdm(fz.namelist()):
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def download_data(_save_path: str, _url: str) -> None:
    '''
    download data from url

    Parameter
    ---------
    * _save_path: path to save the downloaded file
    * _url: url for download
    '''
    try:
        Request.urlretrieve(_url, _save_path)
        return True
    except:
        print('\nError when retrieving the URL:\n{}'.format(_url))
        return False

class BasicDataset(VisionDataset):
    def __init__(self, filepaths: Iterable, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        '''
        Initialize a Pytorch vision dataset from a list of files

        Parameters
        ------------
        * filepaths: a list of paths, each path is an image file
        * transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        * transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        * target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        '''
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        self.transform = transform
        self.target_transform = target_transform

        self.data, self.targets = self.load_images(filepaths)

    def load_images(self, filepaths: list['Path']) -> list:
        '''
        load images to memory with given filepaths.
        class name contained in the path name.

        Parameters
        ----------
        * filepaths: list of pathlib.Path, image path, can be read by Image.open()

        Return
        ----------
        * data: ndarray, image array
        * targets: list of str, image labels
        '''
        data = []
        targets = []
        class_counter = Counter()
        for filepath in tqdm(filepaths):
            class_name = filepath.parent.name
            class_counter[class_name] += 1
            img = Image.open(filepath)
            data.append([np.array(img)])
            targets.extend([class_name])

        self.class_to_idx = {_class: i for i, _class in enumerate(class_counter.keys())}
        # data = np.asarray(data)
        data = np.concatenate(data)
        targets = [self.class_to_idx[x] for x in targets]
        return data, targets


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


class EuroSat(VisionDataset):
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
        'Residential', 'River', 'SeaLake'
    ]
    NUM_CLASSES = 10
    
    # eurosat's stats
    MEAN = (0.3444, 0.3803, 0.4078)  #(0.4914, 0.4822, 0.4465),  #
    STD = (0.2037, 0.1366, 0.1148)  #}, # (0.2471, 0.2435, 0.2616),  # 
    

    def __init__(
        self,
        data_dir: str,
        n_test: float,
        n_val: float,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        '''
        EuroSat

        Parameter
        ----------
        * data_dir: data folder directory 
        * transform: Optional, if none, standard normalization will be applied.
        * target_transform: Optional
        '''
        super(EuroSat, self).__init__(data_dir, transform=transform, target_transform=target_transform)

        # # load images
        rootdir = Path(hydra.utils.get_original_cwd())
        if not os.path.exists(rootdir / data_dir):
            logger.info('Download & unzip...')
            download_data(rootdir / 'data/EuroSAT_RGB.zip', 'http://madm.dfki.de/files/sentinel/EuroSAT.zip')
            unzip_file(rootdir / 'data/EuroSAT_RGB.zip', rootdir + '/data/EuroSAT_RGB')

        logger.info('---------- train/val/test split -----------')
        self.splited_files = self.get_splited_files(rootdir/data_dir, n_val, n_test)


    def get_dataset(self, name:str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> 'VisionDataset':
        '''
        Given a list of file paths, generate Pytorch dataset.

        Parameters
        ----------
        * name: str, train or val or test
        * transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        * target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        '''
        if name == 'test': 
            filepaths = self.splited_files[0]
        elif name == 'val':
            filepaths = self.splited_files[1]
        else:
            filepaths = self.splited_files[-1]

        # default transform
        if transform is None:
            transform = T.Compose([T.ToTensor(), T.Normalize(mean=self.MEAN, std=self.STD)])
        return BasicDataset(filepaths, transform=transform, target_transform=target_transform)

    def get_splited_files(self, data_dir: 'Path', n_val: float, n_test: float) -> list:
        '''
        Split train, validation, test samples according to the ratio.
        The split is batch balanced because it loops by classes and then split inside each class.

        Parameters
        ----------
        * data_dir: data directory contains all samples
        * n_test: the ratio of test samples in total samples
        * n_val: the ratio of validation samples in total samples

        Return
        ----------
        list[
            list[str], # test file paths
            list[str], #val file paths
            list[str] #train file paths
        ]
        '''
        assert isinstance(n_val, float), f'n_val must be float, receiced: {type(n_val)}'
        assert isinstance(n_test, float), f'n_test must be float, receiced: {type(n_test)}'

        splited_files = [[], [], []]  # train, val, test

        for lc in self.CLASSES:
            filepaths = sorted((data_dir / lc).rglob('*.jpg'))  ##!!for reproducibility, sort them!
            # random.Random(50).shuffle(filepaths)  # * independent random seed
            lc_len = len(filepaths)
            sizes = list(map(lambda x: int(x * lc_len), [n_test, n_val, 1 - n_test - n_val]))
            for i, (end, length) in enumerate(zip(accumulate(sizes), sizes)):
                splited_files[i].extend(filepaths[end - length:end])

        return splited_files



    def sampling_to_plot_LR(self, noise_size, **kwargs) -> 'Data.DataLoader':
        '''
        Intended for the visualization of linear regions. Not used...

        '''
        # idx = np.random.permutation(len(self.data))
        # subset = Data.Subset(self, idx[:noise_size])
        noise = []
        generator = torch.Generator()
        generator.manual_seed(10)  # fix the grid data for lr counting
        for i in range(3):
            noise.append(
                torch.normal(mean=self.MEAN[i],
                             std=self.STD[i],
                             size=(noise_size, 1, 64, 64),
                             generator=generator))
        noise = torch.cat(noise, dim=1)

        noise_label = torch.zeros(
            len(noise)) * -1  #np.random.randint(0, 9, size=len(noise))  #TODO: how to set the label of noise?
        noise_label = noise_label.long()
        dataset = Data.TensorDataset(noise, noise_label)
        loader = Data.DataLoader(dataset, **kwargs)
        return loader
