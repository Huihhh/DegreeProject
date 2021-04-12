import wandb
from utils.ema import EMA
from utils.get_signatures import get_signatures
from utils.utils import AverageMeter, accuracy
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, Average
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.utils import setup_logger
from ignite.contrib.engines import common
from ignite.contrib.handlers.wandb_logger import *
from ignite.contrib.handlers.tensorboard_logger import WeightsHistHandler

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
from utils.utils import AverageMeter


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
    ac = lambda x: F.relu(x-outer_r) + F.relu(inner_r-x)
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

class Experiment(object):
    def __init__(self, model, dataset, CFG, plot_sig=False) -> None:
        super().__init__()
        self.dataset = dataset
        self.CFG = CFG.EXPERIMENT
        self.config = CFG #TODO: one object
        # used Gpu or not
        self.use_gpu = self.CFG.use_gpu
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.CFG.use_gpu else 'cpu')
        self.model = model.to(self.device)

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.CFG.wdecay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.Adam(grouped_parameters, lr=self.CFG.optim_lr,)
                                #    momentum=self.CFG.optim_momentum, nesterov=self.CFG.used_nesterov)
        steps_per_epoch = np.ceil(self.dataset.n_train / CFG.DATASET.batch_size) # eval(self.CFG.steps_per_epoch)
        total_training_steps = self.CFG.n_epoch * steps_per_epoch
        warmup_steps = self.CFG.warmup * steps_per_epoch
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_training_steps)
        self.init_criterion()
        self.create_trainer()
        self.to_save = {
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'trainer': self.trainer,
            # 'ema_state_dict': self.ema_model.shadow if self.CFG.ema_used else None
        }

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

        if self.CFG.save_model:
            self.save_checkpoints()

    def init_criterion(self):
        inner_r = self.config.DATASET.boundary_w - self.config.DATASET.width
        outer_r = 1 - self.config.DATASET.width #TODO: these two values are only based on the circle data
        self.criterion = lambda pred, y: torch.nn.BCELoss()(
            pred, y) + self.CFG.dis_reg * disReg(self.model, inner_r, outer_r)

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = batch[0].to(self.device), batch[1].to(self.device).float()
        y_pred = self.model.forward(x)
        loss = self.criterion(y_pred, y[:, None])
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        y_pred = torch.where(y_pred > self.CFG.TH,
                             torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
        acc = accuracy(y_pred, y)
        return {
            'loss': loss.item(),
            'acc': acc
        }

    def validation_step(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = batch[0].to(self.device), batch[1].to(self.device).float()
            y_pred = self.model.forward(x)
            return y_pred, y[:, None]

    def create_trainer(self):
        log_format = "[%(asctime)s][%(module)s.%(filename)s][%(levelname)s] - [%(name)s] %(message)s"
        # trainer
        trainer = Engine(lambda engine, batch: self.train_step(engine, batch))
        trainer.logger = setup_logger('trainer', format=log_format)

        def output_transform_train(out, name):
            return out[name]

        for name in ['loss', 'acc']:
            Average(output_transform=partial(
                output_transform_train, name=name)).attach(trainer, name)

        # evaluator
        evaluator = Engine(
            lambda engine, batch: self.validation_step(engine, batch))
        evaluator.logger = setup_logger("evaluator", level=30)

        def output_transform_val(output):
            y_pred = torch.where(output[0] > self.CFG.TH,
                                 torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
            y = output[1]
            return y_pred, y

        acc = Accuracy(output_transform=output_transform_val)
        ls = Loss(self.criterion)
        ls.attach(evaluator, 'val_loss')
        acc.attach(evaluator, 'val_acc')


        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation_raw(engine):
            # log training results
            metrics = engine.state.metrics
            logger.info(f"[train] train: train_loss={metrics['loss']} train_acc={metrics['acc']}")
            # run validation
            logger.info('======== Validating on original model ========')
            evaluator.run(self.dataset.val_loader)
            metrics = evaluator.state.metrics
            logger.info(
                f"[raw] validation: val_loss={metrics['val_loss']} val_acc={metrics['val_acc']}")

        @trainer.on(Events.COMPLETED)
        def test(engine):
            logger.info("======= Testing =======")
            evaluator.run(self.dataset.test_loader)
            metrics = evaluator.state.metrics
            logger.info(
                f"[raw] Testing: test_loss={metrics['val_loss']} test_acc={metrics['val_acc']}")

        # set up TB logger & Wandb logger
        tb_logger = self.setup_tb_logger(trainer, evaluator)


        if self.CFG.plot_every > 0:
            def custom_event_filter(trainer, event):
                if event in range(10) or event % self.CFG.plot_every == 0:
                    return True
                return False
            events = Events.EPOCH_STARTED(event_filter=custom_event_filter)
            @trainer.on(events)
            def plot_signatures(engine):  # TODO: EPOCH_STARTED?
                total_regions, red_regions, blue_regions, boundary_regions = self.plot_signatures(engine.state.epoch)
                engine.state.metrics.update({
                    'total_regions': total_regions,
                    'red_regions_count': red_regions['count'],
                    'blue_regions_count': blue_regions['count'],
                    'boundary_regions_count': boundary_regions['count'],
                    'red_regions_ratio': red_regions['count'] / (red_regions['area']+1e-6),
                    'blue_regions_ratio': blue_regions['count'] / (blue_regions['area']+1e-6),
                    'boundary_regions_ratio': boundary_regions['count'] / (boundary_regions['area']+1e-6),
                })

                if self.CFG.ema_used:
                    self.ema_model.apply_shadow()
                    logger.info("[EMA] apply shadow")
                    self.plot_signatures(engine.state.epoch)
                    self.ema_model.restore()
                    logger.info("[EMA] restore ")
            tb_logger.attach_output_handler(
                evaluator,
                event_name=events,
                tag="Linear_regions/count",
                metric_names=["total_regions", "red_regions_count", "blue_regions_count", "boundary_regions_count"],
                global_step_transform=global_step_from_engine(trainer),
            )
            tb_logger.attach_output_handler(
                evaluator,
                event_name=events,
                tag="Linear_regions/divided_by_area",
                metric_names=["total_regions", "red_regions_ratio", "blue_regions_ratio", "boundary_regions_ratio"],
                global_step_transform=global_step_from_engine(trainer),
            )
                    # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=events,
                log_handler=WeightsHistHandler(self.model)
            )

        if self.CFG.early_stop:
            def score_function(engine):
                val_loss = engine.state.metrics['val_loss']
                return -val_loss
                
            handler = EarlyStopping(
                patience=10, score_function=score_function, min_delta=0.0001, cumulative_delta=True, trainer=trainer)
            # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
            evaluator.add_event_handler(Events.COMPLETED, handler)

        if self.CFG.ema_used:
            ema_evaluator = Engine(
                lambda engine, batch: self.validation_step(engine, batch))
            ema_evaluator.logger = setup_logger('ema evaluator', level=30)
            val_acc = Accuracy(output_transform=output_transform_val)
            val_acc.attach(ema_evaluator, 'ema_val_acc')
            val_loss = Loss(self.criterion)
            val_loss.attach(ema_evaluator, 'ema_val_loss')

            self.wandb_logger.attach_output_handler(
                ema_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="ema_validation",
                metric_names='all',
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            @trainer.on(Events.ITERATION_COMPLETED)
            def update_ema_params(engien):
                self.ema_model.update_params()

            @trainer.on(Events.EPOCH_COMPLETED)
            def run_validation_ema(engien):
                logger.info('======== Validating on EMA model ========')
                self.ema_model.update_buffer()
                logger.info("[EMA] update buffer()")
                self.ema_model.apply_shadow()
                logger.info("[EMA] apply shadow")
                # validating
                ema_evaluator.run(self.dataset.val_loader)
                metrics = ema_evaluator.state.metrics
                logger.info(
                    f"[EMA] validation: val_loss={metrics['ema_val_loss']} val_acc={metrics['ema_val_acc']}")
                # restore the params
                self.ema_model.restore()
                logger.info("[EMA] restore ")

            @trainer.on(Events.COMPLETED)
            def testing_ema(engine):
                logger.info('======== Testing on EMA model ========')
                self.ema_model.apply_shadow()
                logger.info("[EMA] apply shadow")
                # validating
                ema_evaluator.run(self.dataset.val_loader)
                metrics = ema_evaluator.state.metrics
                logger.info(
                    f"[EMA] validation: val_loss={metrics['ema_val_loss']} val_acc={metrics['ema_val_acc']}")
                # restore the params
                self.ema_model.restore()
                logger.info("[EMA] restore ")

        
        if self.CFG.debug:
            @trainer.on(Events.EPOCH_COMPLETED(every=10))
            def plot_weight_norm_per_layer(engine):
                for name, param in self.model.named_parameters():
                    self.wandb_logger._wandb.log({f'parameters/norm_{name}': LA.norm(param)})


        self.trainer = trainer

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
            fig, ax = plt.subplots(2, 2, sharex='col',sharey='row')
            ax = ax.flatten()
            confidence = net_out.reshape(
                self.grid_labels.shape).detach().cpu().numpy()
            ax0 = ax[0].scatter(xx, yy, c=confidence, vmin=0, vmax=1)
            ax[0].set_title('confidence map')
            fig.colorbar(ax0, ax=ax[0])
            c=1
        else:
            fig, ax = plt.subplots(3, 1, sharex='col',sharey='row')
            ax = ax.flatten()
            c=0
            plt.rcParams['figure.figsize'] = (4.0, 8.0)


        plt.tight_layout(pad=1)
        kwargs = dict(
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
        )
        for lables, name in zip([pseudo_label.squeeze(), grid_labels], [ 'pseudo_label', 'true_label']):
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
        ax[-1].set(xlim=[xx.min(), xx.max()], ylim=[yy.min(), yy.max()], aspect=1)
        ax[-1].set_title('true label')


        self.wandb_logger._wandb.log({f'LinearRegions/epoch{epoch}': wandb.Image(fig)}, commit=False)
        plt.close(fig)

        return total_regions, red_regions, blue_regions, boundary_regions

    def setup_tb_logger(self, trainer, evaluator):
        evaluators = {'training': trainer, "validation": evaluator}
        tb_logger = common.setup_tb_logging(
            output_path='summaries',
            trainer=trainer,
            optimizers=self.optimizer,
            evaluators=evaluators,
            log_every_iters=43,#TODO: automatically set
        )

        if self.CFG.dryrun:
            os.environ['WANDB_MODE'] = 'dryrun'
        self.wandb_logger = WandBLogger(
            project=self.CFG.wandb_project,
            name=self.CFG.name,
            config=self.config,
            sync_tensorboard=True,
            # tensorboard=tb_logger
        )
        self.wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag='training',
            metric_names='all',
            global_step_transform=lambda *_: trainer.state.iteration,
        )
        self.wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names='all',
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
        self.wandb_logger.attach_opt_params_handler(
            trainer,
            event_name=Events.ITERATION_STARTED,
            optimizer=self.optimizer,
            param_name='lr'  # optional
        )

        return tb_logger
    
    def run(self):
        if self.CFG.resume:
            # optionally resume from a checkpoint
            self.resume_model()

        self.trainer.run(self.dataset.train_loader, max_epochs=self.CFG.n_epoch)

    def save_checkpoints(self):  # TODO: register the event using @trainer.on()

        handler = Checkpoint(
            self.to_save,
            DiskSaver('checkpoints', create_dir=True),
            n_saved=None,
            global_step_transform=lambda *_: self.trainer.state.epoch
        )
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.CFG.save_every), handler)
        self.trainer.add_event_handler(
            Events.COMPLETED, handler)

    def load_model(self, mdl_fname):
        print(mdl_fname)
        print(os.getcwd())
        if self.use_gpu:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(mdl_fname, map_location=self.device)
        else:
            checkpoint = torch.load(mdl_fname)
        Checkpoint.load_objects(to_load=self.to_save, checkpoint=checkpoint)
        # self.model.load_state_dict(checkpoint['state_dict'])
        # self.resumed_epoch = checkpoint['epoch']
        # self.model.eval()
        logger.info("Loading previous model")

    def resume_model(self):
        if self.CFG.resume:
            if os.path.isfile(self.CFG.resume_checkpoints):
                print("=> loading checkpoint '{}'".format(
                    self.CFG.resume_checkpoints))
                logger.info("==> Resuming from checkpoint..")
                if self.use_gpu:
                    # Map model to be loaded to specified single gpu.
                    checkpoint = torch.load(
                        self.CFG.resume_checkpoints, map_location=self.device)
                else:
                    checkpoint = torch.load(self.CFG.resume_checkpoints)
                Checkpoint.load_objects(
                    to_load=self.to_save, checkpoint=checkpoint)


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
        experiment = Experiment(model, dataset, CFG)
        experiment.run(dataset.train_loader, dataset.val_loader, dataset.test_loader)

    main()
