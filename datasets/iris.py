from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from sklearn.model_selection import train_test_split
import plotly.express as px

import torch.utils.data as Data
from utils.utils import get_torch_dataset

logger = logging.getLogger(__name__)

class Iris(object):
    def __init__(self, CFG) -> None:
        super().__init__()
        self.CFG = CFG
        self.seed = CFG.seed
        self.n_train = CFG.n_train
        self.n_val = CFG.n_val
        self.n_test = CFG.n_test
        self.batch_size = CFG.batch_size
        self.num_workers = CFG.num_workers

    def make_iris(self, species_id):
        df = px.data.iris()
        df = df.loc[df['species_id'].isin(species_id)]
        data = df[['sepal_length', 'sepal_width',  'petal_length', 'petal_width']]
        normalized_data=(data-data.mean())/data.std()
        targets = (df[['species_id']].values > min(species_id)).astype(int)
        return normalized_data.values, targets

    def get_dataset(self):
        data, targets = self.make_iris(self.CFG.species_id)
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=self.n_test, random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.n_val, random_state=self.seed)
        self.trainset = get_torch_dataset([X_train, y_train])
        self.valset = get_torch_dataset([X_val, y_val])
        self.testset = get_torch_dataset([X_test, y_test])

    def get_dataloader(self):
        kwargs = dict(batch_size=self.batch_size,
                      num_workers=self.num_workers, pin_memory=True, drop_last=False)
        self.train_loader = Data.DataLoader(
            self.trainset, shuffle=True, **kwargs)
        self.val_loader = Data.DataLoader(self.valset, **kwargs)
        self.test_loader = Data.DataLoader(self.testset, **kwargs)

    def make_points_to_plot_LR(self, use_grid=False):
        return self.make_iris(self.CFG.species_id)
        
    
if __name__ == '__main__':
    import sys, os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    print(sys.path)

    @hydra.main(config_path='../config/dataset/', config_name='iris')
    def main(CFG: DictConfig):
        logger.info(OmegaConf.to_yaml(CFG))
        import time
        t1 = time.time()
        CFG = OmegaConf.structured(OmegaConf.to_yaml(CFG))
        CFG.DATASET.seed = 0
        sat = Iris(CFG.DATASET)
        print(time.time() - t1)
    main()
