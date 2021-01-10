import torch
from torch import optim, nn
import logging
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

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
        self.use_gpu = cfg.use_gpu 
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')

        self.trainer = create_supervised_trainer(self.model,self.optimizer, self.loss_func)
        val_metrics = {
            'accuracy': Accuracy(),
            'nll': Loss(self.loss_func)
        }
        self.evaluator = create_supervised_evaluator(self.model, metrics=val_metrics)
        self.register_events()


    def init_trainer(self):
        pass

    def register_events(self):
        trainer = self.trainer
        evaluator = self.evaluator

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
            evaluator.run(self.dataset.val_loader)
            metrics = evaluator.state.metrics
            print("Validation Results - Epoch[{}] Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))


    def forward(self, inputs):
        return self.model(inputs)

    
    def train_step(self):
        logger.info('***** Running training *****')

    def run(self):
        self.trainer.run(self.dataset.train_loader, max_epochs=20) #TODO: max_epochs


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