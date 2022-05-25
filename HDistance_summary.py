from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import wandb
from utils import flat_omegadict, set_random_seed

 
@hydra.main(config_path='./config', config_name='sampleEfficiency')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)

    # # For reproducibility, set random seed
    set_random_seed(CFG.seed)

    api = wandb.Api()
    runs = api.runs('degree-project-Spiral', filters={
        'config.fc_winit_name': CFG.MODEL.fc_winit.name, 
        'config.EXPERIMENT_n_epoch': CFG.EXPERIMENT.n_epoch,
        'config.MODEL_h_nodes': list(CFG.MODEL.h_nodes)
        })
    
    x_axis = 'epoch'
    wandb.init(job_type='Hamming Distance', project=CFG.EXPERIMENT.wandb_project, config=config, name=CFG.EXPERIMENT.name)
    for run in runs:
        hdis = run.history(x_axis=x_axis, keys=['avg. Hamming distance'])
        acc = run.history(x_axis=x_axis, keys=['train.acc'], samples=1000)
        acc = acc[acc.epoch.isin(hdis.epoch)]
        data = wandb.Table(data=acc.merge(hdis), columns=["avg. Hamming distance", "train.acc"])
        scatter = wandb.plot.scatter(data, "avg. Hamming distance", "train.acc", title="avg. Hamming distance vs train.acc")
        wandb.log({'scatter': scatter})
        
    wandb.finish()






if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
