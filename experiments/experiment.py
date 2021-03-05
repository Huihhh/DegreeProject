from utils.get_signatures import get_signatures
from utils.utils import AverageMeter, accuracy
import time
from os.path import exists
from os import mkdir
import logging
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from collections import Counter

import os
import sys
import math
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
        self.init_lossfc()

        # init grid points to plot linear regions
        if CFG.plot_every > 0 or plot_sig:
            self.grid_points, self.grid_labels = dataset.get_decision_boundary()

            # dir to save plot
            self.save_folder = Path('LinearRegions/')
            if not exists(self.save_folder):
                mkdir(self.save_folder)

    def init_lossfc(self):
        bias_reg_loss = AverageMeter()
        if self.CFG.bdecay > 0:
            bdecay_m = eval(self.CFG.bdecay_m) if isinstance(self.CFG.bdecay_m, str) else self.CFG.bdecay_m
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    #TODO: try other loss regularization methods
                    bias_reg_loss.update(torch.sum(torch.abs(torch.abs(param) - bdecay_m)))
        self.loss_func = lambda pred, y: torch.nn.BCELoss()(pred, y) + self.CFG.bdecay * bias_reg_loss.avg

    def train_step(self):
        logger.info("----- Running training -----")
        train_losses_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        # start traning
        self.model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(self.dataset.train_loader):
            # forward
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).float()
            y_pred = self.model(batch_x)
            
            # compute loss and accuracy
            loss = self.loss_func(y_pred, batch_y[:, None])
            y_pred = torch.where(y_pred > self.CFG.TH,
                                 torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
            acc = accuracy(y_pred, batch_y)

            # compute gradient and backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # update recording
            train_losses_meter.update(loss.item())
            train_acc_meter.update(acc)
        return train_losses_meter.avg, train_acc_meter.avg

    def valation_step(self):
        logger.info("----- Running valation -----")
        val_losses_meter = AverageMeter()
        val_acc_meter = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(self.dataset.val_loader):
                # forward
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).float()
                y_pred = self.model(batch_x)

                # compute loss and accuracy
                loss = self.loss_func(y_pred, batch_y[:, None])
                y_pred = torch.where(y_pred > self.CFG.TH, torch.tensor(
                    1.0).to(self.device), torch.tensor(0.0).to(self.device))
                acc = accuracy(y_pred, batch_y)

                # update recording
                val_losses_meter.update(loss.item(), batch_x.shape[0])
                val_acc_meter.update(acc)
        return val_losses_meter.avg, val_acc_meter.avg

    def testing(self):
        logger.info("***** Running testing *****")
        start = time.time()
        batch_time_meter = AverageMeter()
        test_losses_meter = AverageMeter()
        top1_meter = AverageMeter()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.test_loader):
                self.model.eval()
                if self.use_gpu:
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device).float()
                # forward
                outputs = self.forward(inputs)
                # compute loss and accuracy
                loss = self.loss_func(outputs, targets[:, None])
                outputs = torch.where(outputs > self.CFG.TH, torch.tensor(
                    1.0).to(self.device), torch.tensor(0.0).to(self.device))
                acc = accuracy(outputs, targets)
                # update recording
                test_losses_meter.update(loss.item(), inputs.shape[0])
                top1_meter.update(acc, inputs.shape[0])
                batch_time_meter.update(time.time() - start)

        test_loss, top1_acc = test_losses_meter.avg, top1_meter.avg
        logger.info(
            "[Testing] testing_loss: {:.4f}, test acc:{}".format(test_loss, top1_acc))

    def plot_signatures(self, epoch_idx):
        xx, yy = self.grid_points[:, 0], self.grid_points[:, 1]
        net_out, sigs_grid, _ = get_signatures(torch.tensor(self.grid_points).float().to(self.device), self.model)
        net_out = torch.sigmoid(net_out)
        pseudo_label = torch.where(net_out.cpu() > self.CFG.TH, torch.tensor(1), torch.tensor(-1)).numpy()
        sigs_grid = np.array([''.join(str(x) for x in s.tolist()) for s in sigs_grid])
        sigs_grid_counter = Counter(sigs_grid)
        total_regions = len(sigs_grid_counter)
        boundary_regions = 0
        grid_labels = self.grid_labels.reshape(-1)
        for key in sigs_grid_counter:
            idx = np.where(sigs_grid == key)
            region_labels = grid_labels[idx]
            ratio = sum(region_labels) / region_labels.size
            if ratio > -0.2 and ratio < 0.2:  # TODO:parameterize threshold, or get the value geometrically?
                boundary_regions += 1
        logger.info(f'[Linear regions] #around the boundary / total: {boundary_regions} / {total_regions}')

        for lables, name in zip([grid_labels, pseudo_label], ['true_label', 'pseudo_label']):
            color_labels = np.zeros(lables.shape)
            for i, key in enumerate(sigs_grid_counter):
                idx = np.where(sigs_grid == key)
                region_labels = lables[idx]
                ratio = sum(region_labels) / region_labels.size
                # if ratio == 1.0 or ratio == -1.0:
                #     color_labels[idx] = ratio + ratio * np.random.random()
                # else:
                color_labels[idx] = (ratio + np.random.random()) / 2

            color_labels = color_labels.reshape(self.grid_labels.shape)
            if self.dataset.CFG.name != 'circles_fill':
                color_labels = color_labels.T
            # random_labels = np.array([sigs_grid_counter[s] for s in sigs_grid]).reshape(self.grid_labels.shape)
            plt.figure(figsize=(10, 10), dpi=125)
            plt.imshow(color_labels,
                       interpolation="nearest",
                       vmax=1.0,
                       vmin=-1.0,
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.get_cmap('bwr'),
                       aspect="auto",
                       origin="lower",
                       alpha=1)
            if self.CFG.plot_points:
                input_points, labels = self.dataset.data
                plt.scatter(input_points[:, 0], input_points[:, 1], c=labels, linewidths=0.5)

            plt.savefig(self.save_folder / f'{name}_epoch{epoch_idx}.png')

        # save confidence map
        if self.CFG.plot_confidence:
            confidence = net_out.reshape(self.grid_labels.shape).detach().cpu().numpy()
            # plt.figure(figsize=(14, 10))
            plt.scatter(xx, yy, c=confidence, vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(self.save_folder / f'confidenc_epoch{epoch_idx}.png')
        return boundary_regions, total_regions

    def fitting(self):
        # initialize tensorboard
        self.swriter = SummaryWriter(log_dir='summaries')

        if self.CFG.resume:
            # optionally resume from a checkpoint
            start_epoch = self.resume_model()
        else:
            start_epoch = 0

        self.map_color = {}
        prev_lr = np.inf
        for epoch_idx in range(start_epoch, self.CFG.n_epoch):
            # get signatures
            if self.CFG.plot_every and epoch_idx == 0:
                boundary_regions, total_regions = self.plot_signatures(epoch_idx)
                self.swriter.add_scalars(
                    'linear_regions', {'total': total_regions, 'boundary': boundary_regions},
                    epoch_idx)

            # training
            start = time.time()
            train_loss, train_acc = self.train_step()
            end = time.time()
            print("epoch {}: use {} seconds".format(epoch_idx, end - start))

            cur_lr = self.optimizer.param_groups[0]['lr']  # self.scheduler.get_last_lr()[0]
            if cur_lr != prev_lr:
                print('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr

            # plot linear regions
            if self.CFG.plot_every and (epoch_idx + 1) % self.CFG.plot_every == 0:
                boundary_regions, total_regions = self.plot_signatures(epoch_idx)
                self.swriter.add_scalars(
                    'linear_regions', {'total': total_regions, 'boundary': boundary_regions},
                    epoch_idx)
            # valating
            val_loss, val_acc = self.valation_step()

            # saving data in tensorboard
            self.swriter.add_scalars('lr', {'Current Lr': cur_lr}, epoch_idx)
            self.swriter.add_scalars(
                'loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch_idx)
            self.swriter.add_scalars(
                'accuracy/', {'train': train_acc, 'val': val_acc}, epoch_idx)

            logger.info('Epoch {}. [Train] time:{} seconds, '
                        'train_loss: {:.4f}, val_loss: {:.4f}, '
                        'train acc: {}, val acc: {}'.format(
                            epoch_idx, end-start, train_loss, val_loss, train_acc, val_acc)
                        )

            # saving model
            if self.CFG.save_model:
                if epoch_idx == 0 or (epoch_idx + 1) % self.CFG.save_every == 0:
                    state = {
                        'epoch': epoch_idx,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }
                    self.save_checkpoint(state, epoch_idx)

        # close tensorboard
        self.swriter.close()

    def save_checkpoint(self, state, epoch_idx):
        saving_checkpoint_file_folder = Path('checkpoints')

        if not exists(saving_checkpoint_file_folder):
            mkdir(saving_checkpoint_file_folder)
        filename = os.path.join(saving_checkpoint_file_folder,
                                '{}_epoch_{}.pth.tar'.format(self.CFG.name, epoch_idx))
        torch.save(state, filename)
        logger.info("[Checkpoints] Epoch {}, saving to {}".format(
            state['epoch'], filename))

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
        # TODO: not test
        """ optionally resume from a checkpoint
        Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
        start_epoch = 0
        if self.CFG.resume:
            if os.path.isfile(self.CFG.resume_checkpoints):
                print("=> loading checkpoint '{}'".format(self.CFG.resume_checkpoints))
                logger.info("==> Resuming from checkpoint..")
                if self.use_gpu:
                    # Map model to be loaded to specified single gpu.
                    checkpoint = torch.load(self.CFG.resume_checkpoints, map_location=self.device)
                else:
                    checkpoint = torch.load(self.CFG.resume_checkpoints)
                start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])

                print("=> loaded checkpoint '{}' (epoch {})".format(self.CFG.resume_checkpoints, checkpoint['epoch']))
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    self.CFG.resume_checkpoints, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.CFG.resume_checkpoints))
                logger.info("=> no checkpoint found at '{}'".format(self.CFG.resume_checkpoints))
        return start_epoch

    def forward(self, inputs):
        return self.model(inputs)

    def run(self):
        self.fitting()


if __name__ == "__main__":
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
        experiment.run()

    main()
