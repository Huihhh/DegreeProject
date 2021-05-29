from collections import defaultdict, Counter
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import optim
from torch import linalg as LA
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import wandb
from sklearn.decomposition import PCA

import numpy as np
import logging
from easydict import EasyDict as edict
from pylab import *
import os
import sys
import itertools

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.utils import acc_topk, hammingDistance, hammingDistance_classwise
from utils.lr_schedulers import get_cosine_schedule_with_warmup
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

    def init_criterion(self):
        def compute_loss(feature, y_pred, y):
            ce_loss = F.cross_entropy(y_pred, y)
            _, sigs, _ = get_signatures(feature, self.model.fcs)
            hreg_same_class, hreg_diff_class = hammingDistance_classwise(sigs.float(), y, self.device)
            h_reg = hreg_same_class / (hreg_diff_class + 1e-6)
            total_loss = ce_loss + self.CFG.lambda_hreg * h_reg
            return {'total_loss': total_loss, 'ce_loss': ce_loss, 'hreg': h_reg }
        if self.CFG.lambda_hreg > 0:
            self.criterion = compute_loss
        else:
            self.criterion = lambda _, y_pred, y: {'total_loss': F.cross_entropy(y_pred, y)}

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.CFG.optim_lr, weight_decay=self.CFG.wdecay)
        #    momentum=self.CFG.optim_momentum, nesterov=self.CFG.used_nesterov)
        if self.CFG.scheduler == 'steplr':
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=self.CFG.step_size, gamma=self.CFG.steplr_gamma),
                'interval': 'epoch',
            }
        else:
            steps_per_epoch = np.ceil(len(self.dataset.trainset[0]) /
                                      self.config.batch_size)  # eval(self.CFG.steps_per_epoch)
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
        features_norm = []
        for i, (batch_x, _) in enumerate(self.dataset.train_loader[0]):
            feature = self.model.resnet18(batch_x.to(self.device)).squeeze().cpu()
            feature_norm = torch.norm(feature, dim=1)
            features_norm.extend(list(feature_norm))
            features.extend(feature.view(-1))
            if i > 19:
                break
        self.log(f'historgram.features_norm', wandb.Histogram(features_norm))
        self.log(f'historgram.features', wandb.Histogram(features))

    def on_train_epoch_start(self) -> None:
        logger.info(f'======== Training epoch {self.current_epoch} ========')
        if self.current_epoch in range(10) or (self.current_epoch + 1) % self.CFG.plot_every == 0:
            self.plot_signatures()

    def training_step(self, batch, batch_idx):
        # x, y = [], []
        # for b in batch:
        #     x.append(b[0])
        #     y.append(b[1])
        # x = torch.cat(x)
        # y = torch.cat(y)
        x, y = batch
        features = self.model.resnet18(x).squeeze()
        y_pred = self.model.fcs(features)
        losses = self.criterion(features, y_pred, y) 
        acc, = acc_topk(y_pred, y)
        for name, metric in losses.items():
            self.log(f'train.{name}', metric.item(), on_step=False, on_epoch=True)
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
                    self.log(f'historgram/{name}', wandb.Histogram(param.detach().cpu().view(-1)))
        if self.CFG.ema_used:
            logger.info(f'======== Validating on EMA model: epoch {self.current_epoch} ========')
            self.ema_model.update_buffer()
            self.ema_model.apply_shadow()
        else:
            logger.info(f'======== Validating on Raw model: epoch {self.current_epoch} ========')

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        features = self.model.resnet18(x).squeeze()
        y_pred = self.model.fcs(features)
        losses = self.criterion(features, y_pred, y) 
        acc, = acc_topk(y_pred, y)
        for name, metric in losses.items():
            self.log(f'val.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('val.acc', acc, on_step=False, on_epoch=True)

        if self.CFG.plot_avg_distance and (self.current_epoch in range(10) or
                                           (self.current_epoch + 1) % self.CFG.plot_every == 0):
            _, dis = compute_distance(features, self.model.fcs)
            dis_x_neurons = torch.mean(dis) * self.model.n_neurons
            self.log('dis_x_neurons', dis_x_neurons, on_step=False, on_epoch=True)
        return acc

    def validation_epoch_end(self, *args, **kwargs):
        if self.CFG.ema_used:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        features = self.model.resnet18(x).squeeze()
        y_pred = self.model.fcs(features)
        losses = self.criterion(features, y_pred, y) 
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
            callbacks.append(EarlyStopping('val.total_loss', min_delta=0.0001, patience=20, mode='min', strict=True))

        trainer = pl.Trainer(
            num_sanity_val_steps=-1,  #Sanity check runs n validation batches before starting the training
            # accelerator="ddp",  # if torch.cuda.is_available() else 'ddp_cpu',
            # gradient_clip_val=1,
            callbacks=callbacks,
            logger=wandb_logger,
            checkpoint_callback=False if self.CFG.debug else checkpoint_callback,
            gpus=-1 if torch.cuda.is_available() else 0,
            max_epochs=self.CFG.n_epoch,
            # gradient_clip_val=1,
            progress_bar_refresh_rate=0)
        trainer.fit(self, self.dataset.train_loader[0], self.dataset.val_loader)
        if self.CFG.debug:
            trainer.test(self, test_dataloaders=self.dataset.test_loader)
        else:
            trainer.test(test_dataloaders=self.dataset.test_loader)

    def plot_signatures(self):
        self.model.eval()
        self.log('epoch', self.current_epoch)
        # dataloader = [self.dataset.train_loader[0]]
        noisy_dataloader = [self.dataset.noise_loader]
        loaders = [[self.dataset.train_loader[0]], noisy_dataloader, [self.dataset.val_loader], [self.dataset.test_loader]]
        for i, name in enumerate(['train', 'noise', 'val', 'test']):
            sigs = []
            labels = []
            for batch in zip(*loaders[i]):
                batch_x = torch.zeros_like(batch[0][0])
                for b in batch:
                    batch_x += b[0]
                batch_y = batch[0][1]
                feature = self.model.resnet18(batch_x.to(self.device)).squeeze()
                _, sig, _ = get_signatures(feature, self.model.fcs)
                sigs.append(sig)
                labels.append(batch_y)
            sigs = torch.cat(sigs, dim=0)
        
            h_distance = hammingDistance(sigs.float(), device=self.device)
            self.log(f'hamming_distance/{name}', h_distance.mean())

            if name in ['train', 'val', 'test']:
                labels = torch.cat(labels, dim=0)
                hdis_same, hdis_diff = hammingDistance_classwise(sigs, labels, self.device)
                # comb = list(itertools.combinations(range(num_class), 2))
                # for i in range(num_class):
                #     comb.append((i, i))
                # same_class = diff_class =  0
                # for class1, class2 in comb:
                #     arr1 = sigs[np.where(labels == class1)].float()
                #     arr2 = sigs[np.where(labels == class2)].float()
                #     h_distance = hammingDistance([arr1, arr2], self.device)
                #     if class1 == class2:
                #         same_class += h_distance
                #     else:
                #         diff_class += h_distance
                    # self.log(f'hamming_distance/class{class1}_class{class2}', h_distance)
            self.log(f'hamming_distance/same_class_{name}', hdis_same)
            self.log(f'hamming_distance/diff_class_{name}', hdis_diff)
            

            # get the unique signature of each region
            sigs = np.array([''.join(str(x) for x in s.tolist()) for s in sigs])
            sigs_unique = list(np.unique(sigs))
            total_regions = defaultdict(int)
            total_regions['density'] = len(sigs_unique)

            # get the mapping of region signature and region index
            region_ids = np.random.permutation(total_regions['density'])
            sigs_to_idx = dict(zip(sigs_unique, region_ids))
            region_counts = Counter(sigs)
            size_counts = Counter(region_counts.values())
            np_histogram = (list(size_counts.values()), [0] + sorted(list(size_counts.keys())))
            self.log(f'LR_count_distrib.{name}',
                     wandb.Histogram(np_histogram=np_histogram))

            for key in sigs_to_idx:
                idx = np.where(sigs == key)
                total_regions['density_region_size_over1'] += 1 if len(idx[0]) > 1 else 0

            logger.info(f"[Linear regions] {name} \n   #total regions: {total_regions['density']} ")
            self.log(f'total_regions_{name}', total_regions)

        self.model.train()

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
