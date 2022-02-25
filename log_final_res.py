# Example run: 
# python log_final_res.py EXPERIMENT.wandb_project=degree-project-Spiral EXPERIMENT.name=sample-efficiency-summary init_methods@MODEL.fc_winit=normal_custom 
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
        # runs =  api.runs(f'ahui/degree-project-Spiral', filters={'config.name': {"$regex": f'sample-efficiency-{init_method}-*'}, 'config.seed': seed})
        runs =  api.runs(f'ahui/degree-project-Spiral', 
                    filters={
                        "config.EXPERIMENT_name": 'sample-efficiency', 
                        'config.seed': seed, 
                        "config.fc_winit_name": config['fc_winit_name']
                    }
        )

        for run in runs:
            summary = run.summary
            train_size = run.config['n_samples']
            wandb.log({
                'train_size': train_size,
                'train.acc': summary['train.acc'],
                'val.acc': summary['val.acc'],
                'test.acc': summary['test.acc'],
                'tain.total_loss': summary['tain.total_loss'],
                'val.total_loss': summary['val.total_loss'],
                'test.total_loss': summary['test.total_loss'],
                '#linear regions': summary['#linear regions'],
                })
        wandb.finish()

if __name__ == '__main__':
    main()
