import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import os
import wandb
from pathlib import Path

from datasets.dataset import Dataset
from utils import flat_omegadict, set_random_seed
from utils.get_signatures import get_signatures
from utils.utils import hammingDistance
from nn_models import *

 
@hydra.main(config_path='./config', config_name='sampleEfficiency')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)

    # # For reproducibility, set random seed
    set_random_seed(CFG.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Dataset(seed=CFG.seed, **CFG.DATASET)
    input_dim = dataset.trainset[0][0].shape[0]
    grid_points, _ = dataset.grid_data
    grid_points = torch.tensor(grid_points).float().to(device)
    untrained_model = MODEL[CFG.MODEL.name](input_dim=input_dim,seed=CFG.seed, **CFG.MODEL)
    untrained_model = untrained_model.to(device)
    _, sigs_grid_0, _ = get_signatures(grid_points, untrained_model)
    sigs_grid_0 = torch.unique(sigs_grid_0, dim=0)

    api = wandb.Api()
    runs = api.runs('test', filters={
        'config.fc_winit_name': CFG.MODEL.fc_winit.name, 
        'config.EXPERIMENT_n_epoch': CFG.EXPERIMENT.n_epoch,
        'config.MODEL_h_nodes': list(CFG.MODEL.h_nodes)
        })
    logged_artifacts = runs[0].logged_artifacts()
    
    # load model
    run = wandb.init(job_type='Hamming Distance', project=CFG.EXPERIMENT.wandb_project, config=config, name=CFG.EXPERIMENT.name)
    for artifact in logged_artifacts:
        # artifact = run.use_artifact(f'test/model_epoch{epoch}:{version}')
        epoch = int(artifact.name.split('epoch')[1].split(':')[0])
        datadir = Path(hydra.utils.get_original_cwd()) / f'artifacts/{artifact.name}'
        artifact.download(root=datadir)
        model = torch.load(datadir / 'model.h5')
        model = model.to(device)

        _, sigs_grid, _ = get_signatures(grid_points, model)
        sigs_grid = torch.unique(sigs_grid, dim=0)
        hdistance = hammingDistance([sigs_grid, sigs_grid_0], device).diag()
        avg_Hdistance = hdistance.mean() # ? mean?
        wandb.log({'epoch': epoch, 'avg. Hamming distance': avg_Hdistance, 'distribution': wandb.Histogram(hdistance.cpu())})






if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
