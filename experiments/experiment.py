import torch, time
from os.path import exists
from os import mkdir
from torch import optim, nn
import logging
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.get_signatures import get_signatures
from utils.utils import AverageMeter, accuracy, randomcolor


logger = logging.getLogger(__name__)


class Experiment(object):
    def __init__(self, model, dataset, cfg) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        params = [{'params': model.parameters(), 'weigh_decay': self.cfg.wdecay}]
        self.optimizer = optim.SGD(params, lr=self.cfg.optim_lr,
                                   momentum=self.cfg.optim_momentum, nesterov=self.cfg.used_nesterov)
        self.loss_func = nn.CrossEntropyLoss()

        # used Gpu or not
        self.use_gpu = cfg.use_gpu 
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')

        # log path
        self.summary_logdir = os.path.join(self.cfg.log_path, 'summaries')


    def train_step(self):
        logger.info("***** Running training *****")
        train_losses_meter = AverageMeter()

        # start traning
        self.model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(self.dataset.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            y_pred = self.model(batch_x)
            loss = self.loss_func(y_pred, batch_y)

            # compute gradient and backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # update recording
            train_losses_meter.update(loss.item())
        return train_losses_meter.avg

    def valation_step(self):
        logger.info("***** Running valation *****")
        val_losses_meter = AverageMeter()
        val_acc_meter = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(self.dataset.val_loader):
                # forward
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                y_pred = self.model(batch_x)

                # compute loss and accuracy
                loss = self.loss_func(y_pred, batch_y)
                acc, = accuracy(y_pred, batch_y)

                # update recording
                val_losses_meter.update(loss.item(), batch_x.shape[0])
                val_acc_meter.update(acc.item())
            return val_losses_meter.avg, val_acc_meter.avg

        

    def plot_signatures(self, epoch_idx):
        signatures, code, plot_data =[], [], []
        for batch_x, _ in self.dataset.train_loader:
            signatures.append(get_signatures(batch_x, self.model)[1])
            plot_data.append(batch_x)
        signatures = torch.cat(signatures, 0)
        plot_data = torch.cat(plot_data, 0).numpy()
        for s in signatures:
            code.append(''.join(str(x) for x in s.tolist()))

        map_color = { c:i for i, c in enumerate(set(code))}
        colors = [randomcolor(map_color[c]) for c in code]
        plt.scatter(plot_data[:, 0], plot_data[:, 1], c=colors)
        plt.title(f'Epoch{epoch_idx}')
        plt.savefig(f'./outputs/scatter_plots/epoch{epoch_idx}.png')

    def fitting(self):
        # initialize tensorboard
        self.swriter = SummaryWriter(log_dir=self.summary_logdir)
        
        if self.cfg.resume:
            # optionally resume from a checkpoint
            start_epoch = self.resume_model()
        else:
            start_epoch = 0

        for epoch_idx in range(start_epoch, self.cfg.n_epoch):
            # get signatures
            if epoch_idx % self.cfg.plot_every == 0:
                self.plot_signatures(epoch_idx)


            # training
            start = time.time()
            train_loss = self.train_step()
            end = time.time()
            print("epoch {}: use {} seconds".format(epoch_idx, end - start))

            # valating
            val_loss, val_acc = self.valation_step()

            # saving data in tensorboard
            self.swriter.add_scalars('train/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch_idx)
            self.swriter.add_scalars('validation/accuracy', {'accuracy': val_acc}, epoch_idx)
            logger.info('Epoch {}. [Train] time:{} seconds, '
            'train_loss: {:.4f}, val_loss: {:.4f}, '
            'val acc: {}'.format(epoch_idx, end-start, train_loss, val_loss, val_acc)
            )


            # saving model
            if self.cfg.save_model:
                if epoch_idx == 0 or (epoch_idx + 1) % self.cfg.save_every == 0:
                    state = {
                        'epoch': epoch_idx,
                        'state_dic': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    self.save_checkpoint(state, epoch_idx)


            # close tensorboard
            self.swriter.close()

    def save_checkpoint(self, state, epoch_idx):
        saving_checkpoint_file_folder = os.path.join(self.cfg.out_model, self.cfg.log_path.split('/')[-1])
        if not exists(saving_checkpoint_file_folder):
            mkdir(saving_checkpoint_file_folder)
        filename = os.path.join(saving_checkpoint_file_folder,'{}_epoch_{}.pth.tar'.format(self.cfg.name, epoch_idx))
        torch.save(state, filename)
        logger.info("[Checkpoints] Epoch {}, saving to {}".format(state['epoch'], filename))



    def forward(self, inputs):
        return self.model(inputs)
        

    def run(self):
        self.fitting()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import os, sys
    sys.path.append(os.getcwd())
    from models.dnn import SimpleNet
    from datasets.syntheticData import Dataset

    @hydra.main(config_name='config', config_path='../config')
    def main(CFG: DictConfig):
        print('==> CONFIG is \n', OmegaConf.to_yaml(CFG), '\n')
        model = SimpleNet()
        dataset = Dataset(1400, 400, 200)
        experiment = Experiment(model, dataset, CFG.EXPERIMENT)
        experiment.run()
        
    main()