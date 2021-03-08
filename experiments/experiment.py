from enum import unique
from numpy import random
from utils.get_signatures import get_signatures
from utils.utils import AverageMeter, accuracy
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.utils import setup_logger
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter

import math
import numpy as np
import logging
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

import os
from os import mkdir
from os.path import exists
from pathlib import Path

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


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
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class Experiment(object):
    def __init__(self, model, dataset, CFG, plot_sig=False) -> None:
        super().__init__()
        self.dataset = dataset
        self.CFG = CFG
        # used Gpu or not
        self.use_gpu = CFG.use_gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() and CFG.use_gpu else 'cpu')
        self.model = model.to(self.device)

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': CFG.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.Adam(grouped_parameters, lr=CFG.optim_lr,)
        #    momentum=self.CFG.optim_momentum, nesterov=self.CFG.used_nesterov)
        steps_per_epoch = eval(CFG.steps_per_epoch)
        total_training_steps = CFG.n_epoch * steps_per_epoch
        warmup_steps = CFG.warmup * steps_per_epoch
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_training_steps)

        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.init_criterion()
        self.create_trainer()

        # init grid points to plot linear regions
        if CFG.plot_every > 0 or plot_sig:
            self.grid_points, self.grid_labels = dataset.get_decision_boundary()

            # dir to save plot
            self.save_folder = Path('LinearRegions/')
            if not exists(self.save_folder):
                mkdir(self.save_folder)

        if CFG.save_model:
            self.save_checkpoints()

    def init_criterion(self):
        if self.CFG.bdecay > 0:
            bdecay_mean = eval(self.CFG.bdecay_mean) if isinstance(self.CFG.bdecay_mean, str) else self.CFG.bdecay_mean
        else:
            bdecay_mean = 0     

        def bias_reg():
            if self.CFG.bdecay_method == 'l1':
                reg_func = lambda x: torch.sum(torch.abs(torch.abs(x) - bdecay_mean))
            else: 
                reg_func = lambda x: (torch.abs(torch.abs(x) - bdecay_mean))**2
            bias_reg_loss = AverageMeter()
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    bias_reg_loss.update(reg_func(param))
            return bias_reg_loss.avg
        self.criterion = lambda pred, y: torch.nn.BCELoss()(pred, y) + self.CFG.bdecay * bias_reg()

    def create_trainer(self):
        def train_step(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, y = batch[0].to(self.device), batch[1].to(self.device).float()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y[:, None])
            y_pred = torch.where(y_pred > self.CFG.TH,
                                 torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
            acc = accuracy(y_pred, y)
            loss.backward()
            self.optimizer.step()
            return loss.item(), acc

        trainer = Engine(train_step)

        def validation_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = batch[0].to(self.device), batch[1].to(self.device).float()
                y_pred = self.model(x)
                return y_pred, y[:, None]

        evaluator = Engine(validation_step)
        evaluator.logger = setup_logger("evaluator", level=30)

        def output_transform(output):
            y_pred = torch.where(output[0] > self.CFG.TH,
                                 torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
            y = output[1]
            return y_pred, y

        acc = Accuracy(output_transform=output_transform)
        ls = Loss(self.criterion)
        acc.attach(evaluator, 'accuracy')
        ls.attach(evaluator, 'loss')

        def custom_event_filter(trainer, event):
            if event in range(10):
                return True
            return False

        if self.CFG.plot_every > 0:
            @trainer.on(Events.EPOCH_STARTED(event_filter=custom_event_filter) | Events.EPOCH_COMPLETED(every=self.CFG.plot_every))
            def plot_signatures(engine):  
                xx, yy = self.grid_points[:, 0], self.grid_points[:, 1]
                net_out, sigs_grid, _ = get_signatures(torch.tensor(
                    self.grid_points).float().to(self.device), self.model)
                net_out = torch.sigmoid(net_out)
                pseudo_label = torch.where(net_out.cpu() > self.CFG.TH, torch.tensor(1), torch.tensor(-1)).numpy()
                sigs_grid = np.array([''.join(str(x) for x in s.tolist()) for s in sigs_grid])
                sigs_grid_counter = Counter(sigs_grid)
                total_regions = len(sigs_grid_counter)

                grid_labels = self.grid_labels.reshape(-1)            

                for lables, name in zip([grid_labels, pseudo_label], ['true_label', 'pseudo_label']):
                    color_labels = np.zeros(lables.shape)
                    random_labels = np.zeros(lables.shape)
                    for i, key in enumerate(sigs_grid_counter):
                        idx = np.where(sigs_grid == key)
                        region_labels = lables[idx]
                        color_labels[idx] = sum(region_labels) / region_labels.size
                        random_labels[idx] = np.random.random()
                        # color_labels[idx] = (ratio + np.random.random()) / 2

                    if name == 'true_label':
                        unique_labels = np.unique(color_labels)
                        boundary_regions = sum((unique_labels>-0.2) & (unique_labels < 0.2))
                        blue_regions = sum(unique_labels <= -0.2)
                        red_regions = sum(unique_labels >= 0.2)
                        logger.info(f'[Linear regions] \
                                #around the boundary: {boundary_regions} \
                                #red region: {red_regions} \
                                #blue region: {blue_regions}\
                                #total regions: {total_regions} ')
                        self.swriter.add_scalars(
                                'linear_regions', 
                                {'total': total_regions, 
                                'boundary': boundary_regions,
                                'red_region': red_regions,
                                'blue_region': blue_regions
                                },
                                engine.state.epoch)
                    color_labels = (color_labels + random_labels) / 2
                    color_labels = color_labels.reshape(self.grid_labels.shape)

                    if self.dataset.CFG.name != 'circles_fill':
                        color_labels = color_labels.T

                    plt.figure(figsize=(10, 10), dpi=125)
                    cmap = mpl.cm.viridis
                    norm = mpl.colors.BoundaryNorm(self.CFG.TH_bounds, cmap.N, extend='both')
                    plt.imshow(color_labels,
                               interpolation="nearest",
                               vmax=1.0,
                               vmin=-1.0,
                               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                               cmap=cmap, #plt.get_cmap('bwr'),
                               norm=norm,
                               aspect="auto",
                               origin="lower",
                               alpha=1)
                    plt.imshow(color_labels,
                               interpolation="nearest",
                               vmax=1.0,
                               vmin=-1.0,
                               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                               cmap=plt.get_cmap('bwr'),
                               aspect="auto",
                               origin="lower",
                               alpha=0.6)
                    plt.colorbar()
                    if self.CFG.plot_points:
                        input_points, labels = self.dataset.data
                        plt.scatter(input_points[:, 0], input_points[:, 1], c=labels, linewidths=0.5)

                    plt.savefig(self.save_folder / f'{name}_epoch{engine.state.epoch}.png')

                # save confidence map
                if self.CFG.plot_confidence:
                    confidence = net_out.reshape(self.grid_labels.shape).detach().cpu().numpy()
                    # plt.figure(figsize=(14, 10))
                    plt.scatter(xx, yy, c=confidence, vmin=0, vmax=1)
                    plt.colorbar()
                    plt.savefig(self.save_folder / f'confidenc_epoch{engine.state.epoch}.png')


        @trainer.on(Events.ITERATION_COMPLETED)
        def update_loss_acc(engine):
            self.train_loss.update(engine.state.output[0])
            self.train_acc.update(engine.state.output[1])

        @trainer.on(Events.EPOCH_STARTED)
        def reset_loss_acc(engine):
            self.train_loss.reset()
            self.train_acc.reset()

        @trainer.on(Events.EPOCH_COMPLETED)
        def write_summaries(engine):
            evaluator.run(self.dataset.val_loader)
            metrics = evaluator.state.metrics
            self.swriter.add_scalars(
                'loss', {'train_loss': self.train_loss.avg, 'val_loss': metrics['loss']}, engine.state.epoch)
            self.swriter.add_scalars(
                'accuracy/', {'train': self.train_acc.avg, 'val': metrics['accuracy']}, engine.state.epoch)

            logger.info('Epoch {}. train_loss: {:.4f}, val_loss: {:.4f}, '
                        'train acc: {:.4f}, val acc: {}'.format(
                            engine.state.epoch, self.train_loss.avg, metrics['loss'],
                            self.train_acc.avg, metrics['accuracy']*100.0)
                        )

        @trainer.on(Events.COMPLETED)
        def test(engine):
            evaluator.run(self.dataset.test_loader)
            metrics = evaluator.state.metrics
            logger.info("======= Testing =======")
            logger.info(
                "[Testing] test loss: {:.4f}, test acc:{}".format(metrics['loss'], metrics['accuracy']*100))

        self.trainer = trainer

    def fitting(self, dataloader):
        # initialize tensorboard
        self.swriter = SummaryWriter(log_dir='summaries')
        if self.CFG.resume:
            # optionally resume from a checkpoint
            self.resume_model()

        self.trainer.run(dataloader, max_epochs=self.CFG.n_epoch)  

    def save_checkpoints(self):
        self.to_save = {
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'trainer': self.trainer}

        handler = Checkpoint(
            self.to_save,
            DiskSaver('checkpoints', create_dir=True),
            n_saved=None,
            global_step_transform=lambda *_: self.trainer.state.epoch
        )
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.CFG.save_every), handler)

    def load_model(self, mdl_fname):
        print(mdl_fname)
        print(os.getcwd())
        if self.use_gpu:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(mdl_fname, map_location=self.device)
        else:
            checkpoint = torch.load(mdl_fname)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.resumed_epoch = checkpoint['epoch']
        self.model.eval()
        logger.info("Loading previous model")

    def resume_model(self):
        if self.CFG.resume:
            if os.path.isfile(self.CFG.resume_checkpoints):
                print("=> loading checkpoint '{}'".format(self.CFG.resume_checkpoints))
                logger.info("==> Resuming from checkpoint..")
                if self.use_gpu:
                    # Map model to be loaded to specified single gpu.
                    checkpoint = torch.load(self.CFG.resume_checkpoints, map_location=self.device)
                else:
                    checkpoint = torch.load(self.CFG.resume_checkpoints)
                Checkpoint.load_objects(to_load=self.to_save, checkpoint=checkpoint)


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
        experiment = Experiment(model, dataset, CFG.EXPERIMENT)
        experiment.fitting(dataset.train_loader)

    main()
