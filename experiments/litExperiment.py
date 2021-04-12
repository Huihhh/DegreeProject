
import wandb
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch import linalg as LA
import torch.nn.functional as F

import math
import numpy as np
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import *
from functools import partial
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.utils import accuracy
from utils.get_signatures import get_signatures
from utils.ema import EMA


logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    """ <Borrowed from `transformers`>
        Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 0,
        after a warmup period during which it increases from 0 to the initial lr set in the optimizer.
        Args:
            optimizer (:class:`~torch.optim.Optimizer`): The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`): The number of steps for the warmup phase.
            num_training_steps (:obj:`int`): The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1): The index of the last epoch when resuming training.
        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        # this is correct
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def disReg(model, inner_r, outer_r):
    acc_b = []
    acc_W = []
    def ac(x): return F.relu(x-outer_r) + F.relu(inner_r-x)
    for name, param in model.named_parameters():
        if 'weight' in name:
            norm_W = torch.sqrt(torch.sum(param**2, dim=1))
            acc_W.append(norm_W)
        elif 'bias' in name:
            norm_b = torch.abs(param)
            acc_b.append(norm_b)
    loss = 0
    for norm_w, norm_b in zip(acc_W, acc_b):
        loss += torch.sum(ac(norm_b / norm_w))
    return loss


class LitExperiment(pl.LightningModule):
    def __init__(self, model, dataset, CFG, plot_sig=False) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.CFG = CFG.EXPERIMENT
        self.config = CFG  # TODO: one object

        self.init_criterion()

        # init grid points to plot linear regions
        if self.CFG.plot_every > 0 or plot_sig:
            if CFG.DATASET.name == 'spiral':
                self.grid_points, self.grid_labels = dataset.get_decision_boundary_spiral()
            else:
                self.grid_points, self.grid_labels = dataset.get_decision_boundary()

        # used EWA or not
        self.ema = self.CFG.ema_used
        if self.ema:
            self.ema_model = EMA(self.model, self.CFG.ema_decay)
            logger.info("[EMA] initial ")

    def init_criterion(self):
        inner_r = self.config.DATASET.boundary_w - self.config.DATASET.width
        # TODO: these two values are only based on the circle data
        outer_r = 1 - self.config.DATASET.width
        self.criterion = lambda pred, y: torch.nn.BCELoss()(
            pred, y) + self.CFG.dis_reg * disReg(self.model, inner_r, outer_r)

    def configure_optimizers(self):
        # optimizer
        # refer to https://github.com/kekmodel/FixMatch-pytorch/blob/248268b8e6777de4f5c8768ee7fc53c4f4c8a13c/train.py#L237
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.CFG.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(grouped_parameters, lr=self.CFG.optim_lr,)
                                #    momentum=self.CFG.optim_momentum, nesterov=self.CFG.used_nesterov)
        steps_per_epoch = np.ceil(self.dataset.n_train / self.config.DATASET.batch_size) # eval(self.CFG.steps_per_epoch)
        total_training_steps = self.CFG.n_epoch * steps_per_epoch
        warmup_steps = self.CFG.warmup * steps_per_epoch
        scheduler = {
            'scheduler': get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps),
            'interval': 'step',
            # 'strict': True,
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

    def on_train_epoch_start(self) -> None:
        if self.current_epoch in range(10) or self.current_epoch % self.CFG.plot_every == 0:
            total_regions, red_regions, blue_regions, boundary_regions = self.plot_signatures(
                self.current_epoch)
            self.log('total_regions', total_regions)
            self.log('red_regions', red_regions)
            self.log('blue_regions', blue_regions)
            self.log('boundary_regions', boundary_regions)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_pred = self.model.forward(x)
        loss = self.criterion(y_pred, y[:, None])
        y_pred = torch.where(y_pred > self.CFG.TH,
                             torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
        acc = accuracy(y_pred, y)
        self.log('train', {'loss': loss.item(), 'acc': acc})
        return loss

    def training_step_end(self, loss, *args, **kwargs):
        if self.CFG.ema_used:
            self.ema_model.update_params()
        return loss

    def on_train_epoch_end(self, outputs) -> None:
        if self.CFG.debug:
            for name, param in self.model.named_parameters():
                self.log(f'parameters/norm_{name}', LA.norm(param))

    def on_validation_epoch_start(self):
        if self.CFG.ema_used:
            logger.info('======== Validating on EMA model ========')
            self.ema_model.update_buffer()
            logger.info("[EMA] update buffer()")
            self.ema_model.apply_shadow()
            logger.info("[EMA] apply shadow")

    def validation_step(self,batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_pred = self.model.forward(x)
        loss = self.criterion(y_pred, y[:, None])
        acc = accuracy(y_pred, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return y_pred, y[:, None]

    def validation_epoch_end(self, *args, **kwargs):
        if self.CFG.ema_used:
            self.ema_model.restore()
            logger.info("[EMA] restore ")

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        y_pred = self.model.forward(x)
        loss = self.criterion(y_pred, y[:, None])
        acc = accuracy(y_pred, y)
        self.log('test', {'loss': loss, 'acc': acc})

    def run(self):
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='degree-project-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )

        wandb_logger = WandbLogger(
            project='degree-project-lit',
            name=self.CFG.name,
            config=self.config,
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stopping = EarlyStopping('val_loss', min_delta=0.0001, patience=10, mode='min', strict=True)

        trainer = pl.Trainer(
            accelerator="ddp", # if torch.cuda.is_available() else 'ddp_cpu',
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            logger=wandb_logger,
            gpus=-1 if torch.cuda.is_available() else 0,
        )
        logger.info("======= Training =======")
        trainer.fit(self, self.dataset.train_loader, self.dataset.val_loader)
        logger.info("======= Testing =======")
        result = trainer.test(test_dataloaders=self.dataset.test_loader)
        logger.info("======= Training done =======")
        logger.info("test", result)

    def plot_signatures(self, epoch):
        name = 'Linear_regions_ema' if self.CFG.ema_used else 'Linear_regions'
        xx, yy = self.grid_points[:, 0], self.grid_points[:, 1]
        net_out, sigs_grid, _ = get_signatures(torch.tensor(
            self.grid_points).float().to(self.device), self.model)
        net_out = torch.sigmoid(net_out)
        pseudo_label = torch.where(
            net_out.cpu() > self.CFG.TH, torch.tensor(1), torch.tensor(-1)).numpy()
        sigs_grid = np.array([''.join(str(x)
                                      for x in s.tolist()) for s in sigs_grid])
        region_sigs = list(np.unique(sigs_grid))
        total_regions = len(region_sigs)
        region_ids = np.random.permutation(total_regions)

        sigs_grid_dict = dict(zip(region_sigs, region_ids))
        base_color_labels = np.array(
            [sigs_grid_dict[sig] for sig in sigs_grid])
        base_color_labels = base_color_labels.reshape(self.grid_labels.shape).T

        grid_labels = self.grid_labels.reshape(-1)
        boundary_regions, blue_regions, red_regions = defaultdict(
            int), defaultdict(int), defaultdict(int)
        if isinstance(self.CFG.TH_bounds, float):
            bounds = [-self.CFG.TH_bounds, self.CFG.TH_bounds]
        else:
            bounds = self.CFG.TH_bounds
        for i, key in enumerate(sigs_grid_dict):
            idx = np.where(sigs_grid == key)
            region_labels = grid_labels[idx]
            ratio = sum(region_labels) / region_labels.size
            if ratio > bounds[1]:
                red_regions['count'] += 1
                red_regions['area'] += region_labels.size
            elif ratio < bounds[0]:
                blue_regions['count'] += 1
                blue_regions['area'] += region_labels.size
            else:
                boundary_regions['count'] += 1
                boundary_regions['area'] += region_labels.size

        logger.info(f"[Linear regions/area] \
            #around the boundary: {boundary_regions['count'] / (boundary_regions['area'] +1e-6)} \
            #red region: {red_regions['count'] / (red_regions['area'] + 1e-6)} \
            #blue region: {blue_regions['count'] / (blue_regions['area'] +1e-6) }\
            #total regions: {total_regions} ")

        # save confidence map
        if self.CFG.plot_confidence:
            fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
            ax = ax.flatten()
            confidence = net_out.reshape(
                self.grid_labels.shape).detach().cpu().numpy()
            ax0 = ax[0].scatter(xx, yy, c=confidence, vmin=0, vmax=1)
            ax[0].set_title('confidence map')
            fig.colorbar(ax0, ax=ax[0])
            c = 1
        else:
            fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')
            ax = ax.flatten()
            c = 0
            plt.rcParams['figure.figsize'] = (4.0, 8.0)

        plt.tight_layout(pad=1)
        kwargs = dict(
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
        )
        for lables, name in zip([pseudo_label.squeeze(), grid_labels], ['pseudo_label', 'true_label']):
            color_labels = np.zeros(lables.shape)
            for i, key in enumerate(sigs_grid_dict):
                idx = np.where(sigs_grid == key)
                region_labels = lables[idx]
                ratio = sum(region_labels) / region_labels.size
                color_labels[idx] = ratio

            color_labels = color_labels.reshape(self.grid_labels.shape).T

            cmap = mpl.cm.bwr
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
            ax[c].imshow(color_labels, cmap=cmap, norm=norm, alpha=1, **kwargs)
            ax[c].imshow(base_color_labels, cmap=plt.get_cmap(
                'Pastel2'), alpha=0.6, **kwargs)
            ax[c].set_title(name)
            ax[c].set(aspect=1)
            c += 1

        # linear regions colored by true labels with sample points
        ax[-1].imshow(color_labels, cmap=cmap, norm=norm, alpha=1, **kwargs)
        ax[-1].imshow(base_color_labels, cmap=plt.get_cmap(
            'Pastel2'), alpha=0.6, **kwargs)
        input_points, labels = self.dataset.data

        ax[-1].scatter(input_points[:, 0],
                       input_points[:, 1], c=labels, s=1)
        ax[-1].set(xlim=[xx.min(), xx.max()],
                   ylim=[yy.min(), yy.max()], aspect=1)
        ax[-1].set_title('true label')

        self.log(f'LinearRegions/epoch{epoch}', wandb.Image(fig))
        plt.close(fig)

        return total_regions, red_regions, blue_regions, boundary_regions

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
    from datasets.syntheticData import Dataset

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG), '\n')
        model = SimpleNet(CFG.MODEL)
        dataset = Dataset(CFG.DATASET)
        experiment = LitExperiment(model, dataset, CFG)
        experiment.run()

    main()
