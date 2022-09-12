import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from sklearn.preprocessing import LabelEncoder

from trainer.summary import Summary
from utils import flat_omegadict, set_random_seed
from datasets.dataset import Dataset

logger = logging.getLogger(__name__)
label_encoder = LabelEncoder()
api = wandb.Api()

@hydra.main(config_path='./config', config_name='summary')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)
    set_random_seed(0)

    # if needs predictions


    summary = Summary(CFG.wandb_project, CFG.run_name, CFG.num_seeds, True)

    if CFG.archs is not None:
        # if config file = summary_decision_boundary, visualize linear regions + boxplot
        assert CFG.num_seeds==20, 'num_seeds must equal 20'
        summary.vis_decision_boundary(CFG.MODEL, CFG.archs)
    elif config['fc_winit_name'] == 'normal':
        # weight variance vs. #linear regions
        filters = {
            'config.fc_winit_name': 'normal',
        }
        summary.metric_boxplot(config, filters)
    elif CFG.DATASET.name == 'eurosat':
        # plot confusion matrix from predictions on testset
        dataset = Dataset(**CFG.DATASET)
        summary.plot_confusion_matrix(dataset.test_dataloader())
    else:
        # if config file = summary_sample_efficiency
        filters = {
            "config.run_name": 'sample-efficiency', 
            "config.fc_winit_name": config['fc_winit_name'],
            "config.MODEL_h_nodes": list(config['MODEL_h_nodes']),
        }
        summary.metric_line_plot(config, filters)




if __name__ == '__main__':
    main()
