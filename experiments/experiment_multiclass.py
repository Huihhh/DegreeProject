from typing import Counter
from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import optim
from torch import linalg as LA
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data.dataloader import DataLoader
import wandb
from sklearn.decomposition import PCA

import numpy as np
import logging
from easydict import EasyDict as edict
from pylab import *
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.utils import AverageMeter, acc_topk, get_cosine_schedule_with_warmup, get_feature_loader
from utils.compute_distance import compute_distance
from utils.get_signatures import get_signatures
from utils.ema import EMA

logger = logging.getLogger(__name__)


class ExperimentMulti(pl.LightningModule):
    def __init__(self, model, dataset, CFG, plot_sig=False) -> None:
        super().__init__()
        self.model = model
        self.model_name = CFG.MODEL.name
        self.dataset = dataset
        self.CFG = CFG.EXPERIMENT
        self.config = edict()
        for value in CFG.values():
            self.config.update(value)

        self.init_criterion()

        # used EWA or not
        self.ema = self.CFG.ema_used
        if self.ema:
            self.ema_model = EMA(self.model, self.CFG.ema_decay)
            logger.info("[EMA] initial ")

        # init pca for visualizing linear regions
        self.pca = PCA(n_components=2)

    def init_criterion(self):
        self.criterion = lambda y_pred, y: {'total_loss': F.cross_entropy(y_pred, y)}

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.CFG.optim_lr, weight_decay=self.CFG.wdecay)
        #    momentum=self.CFG.optim_momentum, nesterov=self.CFG.used_nesterov)
        steps_per_epoch = np.ceil(len(self.dataset.trainset) / self.config.batch_size)  # eval(self.CFG.steps_per_epoch)
        total_training_steps = self.CFG.n_epoch * steps_per_epoch
        warmup_steps = self.CFG.warmup * steps_per_epoch
        scheduler = {
            'scheduler': get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps),
            'interval': 'step',
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model.forward(x)

    def on_post_move_to_device(self):
        super().on_post_move_to_device()
        # used EWA or not
        # init ema model after model moved to device
        if self.CFG.ema_used:
            self.ema_model = EMA(self.model, self.CFG.ema_decay)
            logger.info("[EMA] initial ")

        features = []
        for i, (batch_x, _) in enumerate(self.dataset.sigs_loader):
            feature = self.model.resnet18(batch_x.to(self.device))
            features.append(feature.squeeze())
        features = torch.cat(features, dim=0)

        if self.CFG.plot_dim is not None:
            self.xy_grid = features[:, self.CFG.plot_dim]
        else:
            self.pca.fit_transform(features.cpu().T)
            self.xy_grid = self.pca.components_.T

    def on_train_epoch_start(self) -> None:
        logger.info(f'======== Training epoch {self.current_epoch} ========')
        if self.current_epoch in range(10) or (self.current_epoch + 1) % self.CFG.plot_every == 0:
            self.plot_signatures()

        features = []
        features_norm = []
        for i, (batch_x, _) in enumerate(self.dataset.train_loader):
            feature = self.model.resnet18(batch_x.to(self.device)).squeeze().cpu()
            feature_norm = torch.norm(feature, dim=1)
            features_norm.extend(list(feature_norm))
            features.extend(feature.view(-1))
            if i > 19:
                break
        self.log(f'historgram.features_norm', wandb.Histogram(features_norm))
        self.log(f'historgram.features', wandb.Histogram(features))

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_pred = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        acc, = acc_topk(y_pred, y)
        for name, metric in losses.items():
            self.log(f'tain.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('train.acc', acc, on_step=False, on_epoch=True)
        return losses['total_loss']

    def training_step_end(self, loss, *args, **kwargs):
        if self.CFG.ema_used:
            self.ema_model.update_params()
        return loss

    def on_validation_epoch_start(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.log(f'parameters/norm_{name}', LA.norm(param))
                if self.CFG.log_weights:
                    wandb.log({
                        f'historgram/{name}': wandb.Histogram(param.detach().cpu().view(-1)),
                        'epoch': self.current_epoch
                    })
        if self.CFG.ema_used:
            logger.info(f'======== Validating on EMA model: epoch {self.current_epoch} ========')
            self.ema_model.update_buffer()
            self.ema_model.apply_shadow()
        else:
            logger.info(f'======== Validating on Raw model: epoch {self.current_epoch} ========')

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        x = self.model.resnet18(x).squeeze()
        y_pred = self.model.fcs(x)
        losses = self.criterion(y_pred, y)
        acc, = acc_topk(y_pred, y)
        for name, metric in losses.items():
            self.log(f'val.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('val.acc', acc, on_step=False, on_epoch=True)

        if self.CFG.plot_avg_distance and (self.current_epoch in range(10) or
                                           (self.current_epoch + 1) % self.CFG.plot_every == 0):
            _, dis = compute_distance(x, self.model.fcs)
            dis_x_neurons = torch.mean(dis) * self.model.n_neurons
            self.log('dis_x_neurons', dis_x_neurons, on_step=False, on_epoch=True)
        return acc

    def validation_epoch_end(self, *args, **kwargs):
        if self.CFG.ema_used:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_pred = self.model.forward(x)
        losses = self.criterion(y_pred, y)
        acc, = acc_topk(y_pred, y)
        self.log('test', {**losses, 'acc': acc})
        return acc

    def run(self):
        wandb_logger = WandbLogger(
            project=self.CFG.wandb_project,
            name=self.CFG.name,
            config=self.config,
        )
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint_callback = ModelCheckpoint(
            monitor='val.total_loss',
            dirpath='checkpoints/',
            filename='degree-project-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor]
        if self.CFG.early_stop:
            callbacks.append(EarlyStopping('val.total_loss', min_delta=0.0001, patience=10, mode='min', strict=True))

        trainer = pl.Trainer(
            num_sanity_val_steps=-1,  #Sanity check runs n validation batches before starting the training
            accelerator="ddp",  # if torch.cuda.is_available() else 'ddp_cpu',
            callbacks=callbacks,
            logger=wandb_logger,
            checkpoint_callback=False if self.CFG.debug else checkpoint_callback,
            gpus=-1 if torch.cuda.is_available() else 0,
            max_epochs=self.CFG.n_epoch,
            # gradient_clip_val=1,
            progress_bar_refresh_rate=0)
        trainer.fit(self, self.dataset.train_loader, self.dataset.val_loader)
        if self.CFG.debug:
            trainer.test(self, test_dataloaders=self.dataset.test_loader)
        else:
            trainer.test(test_dataloaders=self.dataset.test_loader)

    def plot_signatures(self):
        self.model.eval()
        def get_sigs(dataloader):
            sigs = []
            for batch_x, _ in dataloader:
                feature = self.model.resnet18(batch_x.to(self.device)).squeeze()
                _, sig, _ = get_signatures(feature, self.model.fcs)
                sigs.append(sig)
            sigs = torch.cat(sigs, dim=0)
            sigs = np.array([''.join(str(x) for x in s.tolist()) for s in sigs])
            return sigs

        sigs_train = get_sigs(self.dataset.train_loader)
        sigs_grid = get_sigs(self.dataset.sigs_loader)

        # sigs_grid = []
        # for feature, in self.sigs_feature_loader:
        #     _, sig, _ = get_signatures(feature.to(self.device), self.model.fcs)
        #     sigs_grid.append(sig)
        # sigs_grid = torch.cat(sigs_grid, dim=0)
        # sigs_grid = np.array([''.join(str(x) for x in s.tolist()) for s in sigs_grid])

        self.model.train()

        # get the unique signature of each region
        region_sigs = list(np.unique(sigs_grid))
        total_regions = defaultdict(int)
        total_regions['density'] = len(region_sigs)
        # get the mapping of region signature and region index
        region_ids = np.random.permutation(total_regions['density'])
        sigs_grid_dict = dict(zip(region_sigs, region_ids))
        # get the number of training points in regions identified by training samples
        sigs_train = Counter(sigs_train)

        for i, key in enumerate(sigs_grid_dict):
            idx = np.where(sigs_grid == key)
            total_regions['non_empty_regions'] += int(sigs_train[key] > 0)
            total_regions['density_region_size_over1'] += 1 if len(idx[0]) > 1 else 0

        logger.info(f"[Linear regions] \n   #total regions: {total_regions['density']} ")
        self.log('epoch', self.current_epoch)
        self.log('total_regions', total_regions)

        # visualize linear regions
        if self.current_epoch in range(10) or (self.current_epoch + 1) % 10 == 0:

            base_color_labels = np.array([sigs_grid_dict[sig] for sig in sigs_grid])
            plt.scatter(self.xy_grid[:, 0],
                        self.xy_grid[:, 1],
                        c=base_color_labels,
                        s=0.2,
                        cmap=plt.get_cmap('Pastel2'))
            self.log(f'LinearRegions/epoch{self.current_epoch}', wandb.Image(plt))
            plt.close()

    def resume_model(self, train_loader, val_loader, test_loader):
        if self.CFG.resume:
            if os.path.isfile(self.CFG.resume_checkpoints):
                logger.info(f"=> loading checkpoint '${self.CFG.resume_checkpoints}'")
                trainer = Trainer(resume_from_checkpoint=self.CFG.resume_checkpoints)
                trainer.fit(self, train_loader, val_loader)
                result = trainer.test(test_dataloaders=test_loader)
                logger.info("test", result)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os
    import sys
    sys.path.append(os.getcwd())
    from models.dnn import SimpleNet
    from datasets.dataset import Dataset

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG), '\n')
        model = SimpleNet(**CFG.MODEL)
        dataset = Dataset(**CFG.DATASET)
        experiment = ExperimentMulti(model, dataset, CFG)
        experiment.run()

    main()
