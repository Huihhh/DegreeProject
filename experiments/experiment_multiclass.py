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

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.utils import acc_topk, hammingDistance, get_hammingdis
from utils.lr_schedulers import get_cosine_schedule_with_warmup
from utils.compute_distance import compute_distance
from utils.get_signatures import get_signatures
from utils.ema import EMA

logger = logging.getLogger(__name__)


class ExperimentMulti(pl.LightningModule): #TODO: make a base class
    def __init__(self, model, dataset, CFG, plot_sig=False) -> None:
        super().__init__()
        self.model = model
        self.model_name = CFG.MODEL.name
        self.dataset = dataset
        self.CFG = CFG.EXPERIMENT
        self.config = CFG

        self.init_criterion()

        # used EWA or not
        self.ema = self.CFG.ema_used
        if self.ema:
            self.ema_model = EMA(self.model, self.CFG.ema_decay)
            logger.info("[EMA] initial ")
        self.lambda_hreg = lambda epoch: (epoch > self.CFG.hreg_start) * self.CFG.lambda_hreg

    def init_criterion(self):
        self.hammingDistance_classwise = get_hammingdis(p=0)
        hammingLoss_func= get_hammingdis(p=1)
        def compute_loss(post_ac, y_pred, y, epoch):
            ce_loss = F.cross_entropy(y_pred, y)
            hreg_same_class, hreg_diff_class = hammingLoss_func(post_ac, y)
            # print(hreg_diff_class, hreg_same_class)
            h_reg = hreg_same_class /(hreg_diff_class + 1e-6)
            total_loss = ce_loss# + self.lambda_hreg(epoch) * h_reg
            return {'total_loss': total_loss, 'ce_loss': ce_loss, 'hreg': h_reg}

        self.criterion = compute_loss

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
            steps_per_epoch = np.ceil(len(self.dataset.trainset) /
                                      wandb.config.batch_size)  # eval(self.CFG.steps_per_epoch)
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

    #     features = []
    #     features_norm = []
    #     for i, (batch_x, _) in enumerate(self.dataset.train_loader[0]):
    #         feature = self.model.resnet(batch_x.to(self.device)).squeeze().cpu()
    #         feature_norm = torch.norm(feature, dim=1)
    #         features_norm.extend(list(feature_norm))
    #         features.extend(feature.view(-1))
    #         if i > 19:
    #             break
    #     self.log(f'historgram.features_norm', wandb.Histogram(features_norm))
    #     self.log(f'historgram.features', wandb.Histogram(features))

    def on_train_epoch_start(self) -> None:
        logger.info(f'======== Training epoch {self.current_epoch} ========')
        if self.current_epoch in range(10) or (self.current_epoch + 1) % self.CFG.plot_every == 0:
            self.plot_signatures()

    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.resnet(x)
        y_pred, pre_ac = self.model.feature_forward(features)
        losses = self.criterion(pre_ac, y_pred, y, self.current_epoch)
        acc, = acc_topk(y_pred, y)
        # log metrics
        for name, metric in losses.items():
            self.log(f'train.{name}', metric.item(), on_step=False, on_epoch=True)
        self.log('train.acc', acc, on_step=False, on_epoch=True)
        return losses['total_loss']

    # def on_after_backward(self) -> None:
    #     for name, param in self.model.fcs.named_parameters():
    #         print(param.grad)

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
        features = self.model.resnet(x)
        y_pred, pre_ac = self.model.feature_forward(features)
        acc, = acc_topk(y_pred, y)
        self.log('val.acc', acc, on_step=False, on_epoch=True)

        if self.CFG.plot_avg_distance and (self.current_epoch in range(10) or
                                           (self.current_epoch + 1) % self.CFG.plot_every == 0):
            _, dis = compute_distance(features.reshape(features.shape[0], -1), self.model.fcs)
            dis_x_neurons = torch.mean(dis) * self.model.n_neurons
            self.log('dis_x_neurons', dis_x_neurons, on_step=False, on_epoch=True)
        return pre_ac, y_pred, y_pred.argmax(1)

    def validation_epoch_end(self, *args, **kwargs):
        pre_ac, scores, y_pred = list(map(torch.cat, np.array(args[0]).T.tolist()))
        y = torch.tensor(self.dataset.valset.targets, device=self.device)
        losses = self.criterion(pre_ac, scores, y, self.current_epoch)
        for name, metric in losses.items():
            self.log(f'val.{name}', metric.item())
        # if self.current_epoch == self.CFG.n_epoch - 1:
        #     # save validation predictions as an artifact
        #     val_res_at = wandb.Artifact("val_pred_" + wandb.run.id, "val_epoch_preds")
        #     columns=["id", "image", "guess", "truth"]
        #     for a in self.dataset.classes:
        #         columns.append("score_" + a)
        #     val_dt = wandb.Table(columns = columns)
        #     for filepath, top_guess, score, truth in zip(self.dataset.valset.imgs, y_pred, scores, y):
        #         img_id = filepath[0].split('/')[-1].split('.')[0]
        #         row = [img_id, wandb.Image(filepath[0]), self.dataset.valset.classes[top_guess], self.dataset.valset.classes[truth]]
        #         for s in score.tolist():
        #             row.append(np.round(s, 4))
        #         val_dt.add_data(*row)
        #     val_res_at.add(val_dt, "val_epoch_res")
        #     wandb.run.log_artifact(val_res_at)
        if self.CFG.ema_used:
            self.ema_model.restore()

    def test_step(self, batch, batch_idx):
        x, y = batch
        features = self.model.resnet(x)
        y_pred, pre_ac = self.model.feature_forward(features)
        # pre_ac_test, pre_ac_val = pre_ac[:n_test], pre_ac[n_test:]
        # losses_test = self.criterion(pre_ac_test, y_pred_test, batch['test'][1], self.current_epoch)
        acc_test, = acc_topk(y_pred, y)
        # self.log('test', {**losses_test, 'acc': acc_test})
        self.log('acc_test', acc_test, on_epoch=True, on_step=False)
        return acc_test

    def on_fit_end(self) -> None:
        config = wandb.config
        model_artifact = wandb.Artifact(
            f'{self.model_name}-{self.CFG.name}', type='model',
            description='model artifact',
            metadata=dict(config)
        )
        with model_artifact.new_file(f'{self.model_name}-{self.CFG.name}' + '.pt', mode='wb') as file:
            torch.save(self.model, file)
        self.logger.experiment.log_artifact(model_artifact)


    def run(self):
        wandb_logger = WandbLogger(
            project=self.CFG.wandb_project,
            name=self.CFG.name,
        )
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        # checkpoint_callback = ModelCheckpoint(
        #     monitor='val.acc',
        #     dirpath='checkpoints/',
        #     filename='degree-project-{epoch:02d}-{val_loss:.2f}',
        #     save_top_k=3,
        #     mode='max',
        # )

        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor]
        if self.CFG.early_stop:
            callbacks.append(EarlyStopping('val.total_loss', min_delta=0.0001, patience=20, mode='min', strict=True))

        trainer = pl.Trainer(
            num_sanity_val_steps=-1,  #Sanity check runs n validation batches before starting the training
            # accelerator="ddp",  # if torch.cuda.is_available() else 'ddp_cpu',
            # gradient_clip_val=5,
            callbacks=callbacks,
            logger=wandb_logger,
            checkpoint_callback=False,# if self.CFG.debug else checkpoint_callback,
            gpus=-1 if torch.cuda.is_available() else 0,
            max_epochs=self.CFG.n_epoch,
            gradient_clip_val=10,
            progress_bar_refresh_rate=0)
        trainer.fit(self, self.dataset)
        trainer.test(self, datamodule=self.dataset)
        # if self.CFG.debug:
        #     trainer.test(self, datamodule=self.dataset)
        # else:
        #     trainer.test(datamodule=self.dataset)

    def plot_signatures(self):
        self.model.eval()
        self.log('epoch', self.current_epoch)
        train_dataloader = self.dataset.train_dataloader()
        noisy_dataloader = [self.dataset.noise_loader, train_dataloader]
        loaders = [[train_dataloader], noisy_dataloader, [self.dataset.val_dataloader()],
                   [self.dataset.test_dataloader()]]
        for i, name in enumerate(['train', 'noise', 'val', 'test']):
            sigs = []
            labels = []
            for batch in zip(*loaders[i]):
                batch_x = torch.zeros_like(batch[0][0])
                for b in batch:
                    batch_x += b[0]
                batch_y = batch[0][1]
                feature = self.model.resnet(batch_x.to(self.device))
                _, sig, _ = get_signatures(feature.reshape(feature.shape[0], -1), self.model.fcs)
                sigs.append(sig)
                labels.append(batch_y)
            sigs = torch.cat(sigs, dim=0)

            # h_distance = hammingDistance(sigs.float(), device=self.device)
            h_distance = F.pdist(sigs.float(), p=0).mean()
            self.log(f'hamming_distance/{name}', h_distance)

            if name in ['train', 'val', 'test']:
                labels = torch.cat(labels, dim=0)
                hdis_same, hdis_diff = self.hammingDistance_classwise(sigs, labels)
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
            self.log(f'LR_count_distrib.{name}', wandb.Histogram(np_histogram=np_histogram))

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
