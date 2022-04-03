# not used
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omegaconf import DictConfig
import hydra
import logging

from models import *


@hydra.main(config_path='./config', config_name='config')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    # logger.info(OmegaConf.to_yaml(CFG))

    # # For reproducibility, set random seed
    if CFG.Logging.seed == 'None':
        CFG.Logging.seed = random.randint(1, 10000)
    random.seed(CFG.Logging.seed)
    np.random.seed(CFG.Logging.seed)
    torch.manual_seed(CFG.Logging.seed)
    torch.cuda.manual_seed_all(CFG.Logging.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if CFG.DATASET.name == 'eurosat':
        input_dim = 512
        model = ResNet(**CFG.MODEL)
    else:
        input_dim = 2

        # build model
        model = MODEL[CFG.MODEL.name](input_dim=input_dim, **CFG.MODEL)
    logger.info("[Model] Building model -- input dim: {}, hidden nodes: {}, out dim: {}".format(
        input_dim, CFG.MODEL.h_nodes, CFG.MODEL.out_dim))

    if CFG.EXPERIMENT.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)

    if isinstance(model, ResNet):
        model = model.fcs
    weights = []
    for name, params in model.named_parameters():
        if 'weight' in name:
            weights.append(params)
    direc = []
    oth_direc = []
    th = 0.087
    for w in weights:
        norm_w = torch.norm(w, dim=1, keepdim=True)
        d = w.mm(w.T) / (norm_w.mm(norm_w.T))
        direc.append(d)
        oth_direc.append(((d < th) * (d > -th)).sum().item())
    print(oth_direc)


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
