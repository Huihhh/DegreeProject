import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from sklearn.preprocessing import LabelEncoder

from experiments.summary import Summary
from utils import flat_omegadict, set_random_seed

logger = logging.getLogger(__name__)
label_encoder = LabelEncoder()
api = wandb.Api()

@hydra.main(config_path='./config', config_name='summary_sample_efficiency')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)
    set_random_seed(0)
    summary = Summary(CFG.wandb_project, CFG.run_name, CFG.num_seeds, True)
    if 'archs' in CFG:
        # if config file = summary_decision_boundary, visualize linear regions + boxplot
        summary.vis_decision_boundary(CFG.MODEL, CFG.archs)
    else:
        # if config file = summary_sample_efficiency
        filters = {
            "config.run_name": 'sample-efficiency', 
            "config.fc_winit_name": config['fc_winit_name'],
            "config.MODEL_h_nodes": list(config['MODEL_h_nodes']),
        }
        summary.plot_accross_runs(config, filters)



if __name__ == '__main__':
    main()
