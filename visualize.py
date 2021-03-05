
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random, glob

from omegaconf import DictConfig, OmegaConf
import hydra

from datasets.syntheticData import Dataset
from models.dnn import SimpleNet
from experiments.experiment import Experiment
import logging

@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))

    # # For reproducibility, set random seed
    if CFG.Logging.seed == 'None':
        CFG.Logging.seed = random.randint(1, 10000)
    random.seed(CFG.Logging.seed)
    np.random.seed(CFG.Logging.seed)
    torch.manual_seed(CFG.Logging.seed)
    torch.cuda.manual_seed_all(CFG.Logging.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # get datasets
    dataset = Dataset(CFG.DATASET)
    # dataset.plot(CFG.EXPERIMENT.out_folder)

    # build model
    model = SimpleNet(CFG.MODEL)
    logger.info("[Model] Building model {} out dim: {}".format(CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    if CFG.EXPERIMENT.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)

    experiment = Experiment(model, dataset, CFG, plot_sig=True)
    if CFG.EXPERIMENT.resume_checkpoints[-3:] == 'tar':
        experiment.load_model(CFG.EXPERIMENT.resume_checkpoints)
        experiment.plot_signatures(epoch_idx=experiment.resumed_epoch)
    else:
        checkpointslist = sorted(
            glob.glob(CFG.EXPERIMENT.resume_checkpoints + '*.pth.tar'))
        def filter_epoch(file):
            for epoch in CFG.EXPERIMENT.resumed_epochs:
                if epoch in file:
                    return True
                else:
                    return False
        checkpointslist = filter(filter_epoch, checkpointslist)
        
        for i in checkpointslist:
            # CONFIG.DATASET.label_num = int(i.split('/')[-2].split("_")[0][3:])
            CFG.EXPERIMENT.resume_checkpoints = i
            # logger.info("[Experiment {}]".format(CONFIG.DATASET.label_num))
            logger.info("[Resume Checkpoints] {}".format(i))
            experiment.load_model(CFG.EXPERIMENT.resume_checkpoints)
            experiment.plot_signatures(epoch_idx=experiment.resumed_epoch)

    logger.info("======= plotting done =======")

if __name__ == '__main__':
    main()

