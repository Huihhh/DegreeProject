import logging
import os
from datetime import datetime
import sys
import torch
import torch.utils.data as Data
from torch.optim.lr_scheduler import LambdaLR
import math

def accuracy(output, target):
    correct_n = (output == target).sum()
    batch_size = target.size(0)
    return correct_n / batch_size


def acc_topk(output, target, topk=(1,)):
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
    print(params.EXPERIMENT.log_path)
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

def get_torch_dataset(np_data):
    X = torch.from_numpy(np_data[0]).float()
    Y = torch.from_numpy(np_data[1]).long()
    return Data.TensorDataset(X, Y)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    """ <Borrowed from `transformers`>
        Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 0,
        after a warmup period during which it increases from 0 to the initial lr set in the optimizer.
        Args:
            optimizer (:class:`~torch.optim.Optimizer`): The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`): The number of steps for the warmup phase.
            num_training_steps (:obj:`int`): The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1): The index of the last epoch when resuming training.
        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        # this is correct
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)