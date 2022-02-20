import os
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl

def set_random_seed(seed):
    if seed == 'None':
        seed = random.randint(1, 10000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False