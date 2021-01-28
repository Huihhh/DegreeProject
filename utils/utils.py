import torch
import logging
import os
from datetime import datetime
import sys
import random
import numpy as np


# def accuracy(output, target):
#     correct_n = (output.view(-1) == target).sum()
#     batch_size = target.size(0)
#     return 100 * correct_n/batch_size


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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

import functools


def convex_hull_graham(points):
    '''
    https://gist.github.com/arthur-e/5cf52962341310f438e96c1f3c3398b8
    Returns points on convex hull in CCW order according to Graham's scan algorithm. 
    By Tom Switzer <thomas.switzer@gmail.com>.
    '''
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    l = functools.reduce(_keep_left, points, [])
    u = functools.reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l

def get_path(region):
    upper_bound, lower_bound = [], []
    for x in sorted(list(set(region[:, 0]))):
        v_line = region[np.where(region[:, 0] == x)]
        v_line.sort(axis=0)
        upper_bound.append(v_line[-1][None, :])
        lower_bound.append(v_line[0][None, :])

    upper_bound = np.concatenate(upper_bound)
    lower_bound = np.concatenate(lower_bound)
    lower_bound = lower_bound[::-1, :]
    path = np.concatenate([upper_bound, lower_bound], 0)
    return path


