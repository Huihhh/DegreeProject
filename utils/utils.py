# import torch
import logging
import os
from datetime import datetime
import sys
import random
import numpy as np
import matplotlib.patches as patches
import matplotlib.path as mpath

def accuracy(output, target):
    correct_n = (output.view(-1) == target).sum()
    batch_size = target.size(0)
    return 100 * correct_n/batch_size


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k
#     Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
    def __init__(self,):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    #
    # def __str__(self):
    #     fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #     return fmtstr.format(**self.__dict__)



def setup_default_logging(params, string = 'Train', default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    output_dir = os.path.join(params.EXPERIMENT.log_path)
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(string)

    def time_str(fmt=None):
        if fmt is None:
            fmt = '%Y-%m-%d_%H:%M:%S'
        return datetime.today().strftime(fmt)

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler(filename=os.path.join(output_dir, f'{time_str()}.log'), mode='a')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def randomcolor(seed=0):
    random.seed(seed)
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color



def get_path(region):
    # region: np.array (N, 2) coordinates of points in region
    if (len(region[:, 0]) - len(set(region[:, 0]))) <=1 and (len(region[:, 1]) - len(set(region[:, 1]))) <= 1:
      return patches.Polygon(region)
    f_path, b_path = [], []
    if len(set(region[:, 0])) < len(set(region[:, 1])):
      idx = 0
      uniqueIdx = set(region[:, 0])
    else: 
      idx = 1 
      uniqueIdx = set(region[:, 1])
    

    for i, v in enumerate(uniqueIdx):
        h_line = [p for p in region if p[idx] == v]
        if i==0 and len(h_line) > 1:
            f_path.append(h_line[0][None, :])
            f_path.append(h_line[-1][None, :])
        elif i == 0 and len(h_line) == 1:
            f_path.append(h_line[0][None, :])
        elif i > 0 and len(h_line) > 1:
            f_path.append(h_line[-1][None, :])
            b_path.append(h_line[0][None, :])
        elif i > 0 and len(h_line) == 1:
            f_path.append(h_line[0][None, :])

    
    doReverse =  idx == 0 
    f_path.sort(key=lambda x: x[0][idx], reverse = doReverse)
    b_path.sort(key=lambda x: x[0][idx], reverse = True)

    if len(b_path) == 0:
        # print(region)
        # print(f_path)
        f_path = np.concatenate(f_path, 0)
        return mpath.Path(f_path)

    f_path = np.concatenate(f_path, 0)
    b_path = np.concatenate(b_path, 0)
    path = np.concatenate([f_path, b_path[::-1, :]], 0)
    return patches.Polygon(path)
