from omegaconf import DictConfig, OmegaConf
import hydra
import logging

from utils import flat_omegadict
import wandb

api = wandb.Api()

@hydra.main(config_path='./config', config_name='sampleEfficiency')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))
    config=flat_omegadict(CFG)
    config['n_epoch']=500
    for seed in range(5):
        config['seed'] = seed
        wandb.init(project=CFG.EXPERIMENT.wandb_project, name=CFG.EXPERIMENT.name, config=config)
        init_method = CFG.EXPERIMENT.name.split('-')[2]
        runs =  api.runs(f'ahui/degree-project-Spiral', filters={'config.name': {"$regex": f'sample-efficiency-{init_method}-*'}, 'config.seed': seed})
        for run in runs:
            summary = run.summary
            train_size = run.config['n_samples']
            wandb.log({
                'train_size': train_size,
                'train.acc': summary['train.acc'],
                'val.acc': summary['val.acc'],
                'tain.total_loss': summary['tain.total_loss'],
                'val.total_loss': summary['val.total_loss'],
                '#linear regions': summary['#linear regions'],
                })
        wandb.finish()

if __name__ == '__main__':
    main()
