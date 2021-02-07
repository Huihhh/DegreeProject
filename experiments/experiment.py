import time
from os.path import exists
from os import mkdir
import logging
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import numpy as np
from collections import Counter

import os, sys, math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.utils import AverageMeter, accuracy
from utils.get_signatures import get_signatures


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
    def __init__(self, model, dataset, cfg) -> None:
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg.EXPERIMENT
        params = [{'params': model.parameters(), 'weigh_decay': self.cfg.wdecay}]
        self.optimizer = optim.Adam(params, lr=self.cfg.optim_lr,)
                                #    momentum=self.cfg.optim_momentum, nesterov=self.cfg.used_nesterov)
        steps_per_epoch = np.ceil(cfg.DATASET.n_train / cfg.DATASET.batch_size)
        total_training_steps = self.cfg.n_epoch * steps_per_epoch
        warmup_steps = self.cfg.warmup * steps_per_epoch
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_training_steps)
        self.loss_func = torch.nn.BCELoss()

        # used Gpu or not
        self.use_gpu = cfg.EXPERIMENT.use_gpu
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
        self.model = model.to(self.device)

        # log path
        self.summary_logdir = os.path.join(self.cfg.log_path, 'summaries')

    def train_step(self):
        logger.info("----- Running training -----")
        train_losses_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        # start traning
        self.model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(self.dataset.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device).float()
            y_pred = self.model(batch_x)
            # print(batch_idx, y_pred)
            loss = self.loss_func(y_pred, batch_y[:, None])
            y_pred =torch.where(y_pred>self.cfg.TH, 1, 0)
            # print('#positive points:', y_pred.sum())
            acc = accuracy(y_pred, batch_y)

            # compute gradient and backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # update recording
            train_losses_meter.update(loss.item())
            train_acc_meter.update(acc.item())
        return train_losses_meter.avg, train_acc_meter.avg

    def valation_step(self):
        logger.info("----- Running valation -----")
        val_losses_meter = AverageMeter()
        val_acc_meter = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(self.dataset.val_loader):
                # forward
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device).float()
                y_pred = self.model(batch_x)

                # compute loss and accuracy
                loss = self.loss_func(y_pred, batch_y[:, None])
                y_pred =torch.where(y_pred>self.cfg.TH, 1, 0)
                acc = accuracy(y_pred, batch_y)

                # update recording
                val_losses_meter.update(loss.item(), batch_x.shape[0])
                val_acc_meter.update(acc.item())
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
                outputs =torch.where(outputs>self.cfg.TH, 1, 0)
                acc = accuracy(outputs, targets)
                # update recording
                test_losses_meter.update(loss.item(), inputs.shape[0])
                top1_meter.update(acc.item(), inputs.shape[0])
                batch_time_meter.update(time.time() - start)

        test_loss, top1_acc = test_losses_meter.avg,top1_meter.avg
        logger.info(
            "[Testing] testing_loss: {:.4f}, test acc:{}".format(test_loss,top1_acc))


    def plot_signatures(self, epoch_idx):

        # plot linear regions with random green colors
        h = 0.01
        xx, yy = np.meshgrid(np.arange(self.dataset.minX, self.dataset.maxX, h),
                             np.arange(self.dataset.minY, self.dataset.maxY, h))
        grid_points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        def compare(point): # TODO: parameterize condition 
            x, y = point
            if x**2 + y**2 <=0.75:
                return -1
            elif x**2 + y**2 >=1.25 and x**2 + y**2 <=2:
                return 1
            else: 
                return 0     

        grid_labels = np.array(list(map(compare, grid_points)))
        sigs_grid = get_signatures(torch.tensor(grid_points).float().to(self.device), self.model)[1]
        sigs_grid = np.array([''.join(str(x) for x in s.tolist()) for s in sigs_grid])
        sigs_grid_counter = Counter(sigs_grid)

        color_labels = np.zeros(grid_labels.shape)
        for i, key in enumerate(sigs_grid_counter):
            idx = np.where(sigs_grid == key)
            region_labels = grid_labels[idx]
            ratio = sum(region_labels) / region_labels.size
            if ratio == 1.0 or ratio == -1.0:
              color_labels[idx] = ratio + ratio * np.random.random()
            else: 
              color_labels[idx] = ratio


        color_labels = color_labels.reshape(xx.shape)
        plt.imshow(color_labels, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.get_cmap('bwr'), aspect="auto", origin="lower", alpha=1)

        # save plot
        if not exists(self.cfg.log_path):
            mkdir(self.cfg.log_path)
        save_folder = os.path.join(self.cfg.log_path, 'scatter_plots/')
        if not exists(save_folder):
            mkdir(save_folder)
        plt.savefig(f'{save_folder}epoch{epoch_idx}.png')


    def fitting(self):
        # initialize tensorboard
        self.swriter = SummaryWriter(log_dir=self.summary_logdir)

        if self.cfg.resume:
            # optionally resume from a checkpoint
            start_epoch = self.resume_model()
        else:
            start_epoch = 0

        self.map_color = {}
        for epoch_idx in range(start_epoch, self.cfg.n_epoch):
            # get signatures
            if self.cfg.plot_every and epoch_idx ==0:
                self.plot_signatures(epoch_idx)

            # training
            start = time.time()
            train_loss, train_acc = self.train_step()
            end = time.time()
            print("epoch {}: use {} seconds".format(epoch_idx, end - start))

            # plot linear regions
            if self.cfg.plot_every and (epoch_idx +1) % self.cfg.plot_every == 0:
                self.plot_signatures(epoch_idx)
            # valating
            val_loss, val_acc = self.valation_step()

            # saving data in tensorboard
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
            if self.cfg.save_model:
                if epoch_idx == 0 or (epoch_idx + 1) % self.cfg.save_every == 0:
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
        saving_checkpoint_file_folder = os.path.join(
            self.cfg.out_model, self.cfg.log_path.split('/')[-1])
        if not exists(saving_checkpoint_file_folder):
            mkdir(saving_checkpoint_file_folder)
        filename = os.path.join(saving_checkpoint_file_folder,
                                '{}_epoch_{}.pth.tar'.format(self.cfg.name, epoch_idx))
        torch.save(state, filename)
        logger.info("[Checkpoints] Epoch {}, saving to {}".format(
            state['epoch'], filename))

    def load_model(self, mdl_fname):
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
        if self.cfg.resume:
            if os.path.isfile(self.cfg.resume_checkpoints):
                print("=> loading checkpoint '{}'".format(self.cfg.resume_checkpoints))
                logger.info("==> Resuming from checkpoint..")
                if self.use_gpu:
                    # Map model to be loaded to specified single gpu.
                    checkpoint = torch.load(self.cfg.resume_checkpoints, map_location=self.device)
                else:
                    checkpoint = torch.load(self.cfg.resume_checkpoints)
                start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])

                print("=> loaded checkpoint '{}' (epoch {})".format(self.cfg.resume_checkpoints, checkpoint['epoch']))
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(self.cfg.resume_checkpoints, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.cfg.resume_checkpoints))
                logger.info("=> no checkpoint found at '{}'".format(self.cfg.resume_checkpoints))
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
        # model = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 1),
        #     nn.Sigmoid()
        # )
        model = SimpleNet(CFG.MODEL)
        dataset = Dataset(CFG.DATASET)
        experiment = Experiment(model, dataset, CFG)
        experiment.run()

    main()
