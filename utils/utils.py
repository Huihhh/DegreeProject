from joblib import parallel_backend
import torch
import torch.utils.data as Data
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from omegaconf.dictconfig import DictConfig


def accuracy(output, target):
    correct_n = (output == target).sum()
    batch_size = target.size(0)
    return correct_n / batch_size


def acc_topk(output, target, topk=(1, )):
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
    def __init__(self, ):
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


def get_torch_dataset(np_data):
    X = torch.from_numpy(np_data[0]).float()
    Y = torch.from_numpy(np_data[1]).float()
    return Data.TensorDataset(X, Y)


def get_feature_loader(dataloader, net, device):
    features = []
    for batch_x, batch_y in dataloader:
        feature = net(batch_x.to(device)).squeeze()
        features.append(feature.cpu())
    features = torch.cat(features)
    feature_dataset = Data.TensorDataset(features)
    feature_dataloader = DataLoader(feature_dataset, batch_size=64, num_workers=4, drop_last=False, pin_memory=True)
    return feature_dataloader

def flat_omegadict(odict, parentKey=None):
    flat = {}
    for key, value in odict.items():
        if isinstance(value, DictConfig) or isinstance(value, dict):
            flat |= flat_omegadict(value, key)
        else:
            newKey = f'{parentKey}.{key}' if parentKey else key
            flat |= {newKey: value}
    return flat

def hammingDistance(arr, device):
    '''
  arr: 0,1 2d array or a list of 0,1 2d array
  '''
    if isinstance(arr, list):
        arr1, arr2 = arr
    else:
        arr1 = arr2 = arr
    n1, m = arr1.shape  ## n1 is the sample size of arr1, m is the feature size
    n2, _ = arr2.shape
    arr_not1 = torch.ones((n1, m), device=device) - arr1
    arr_not2 = torch.ones((n2, m), device=device) - arr2
    arr_ones = arr1.mm(arr2.T)  # count the positions of both ones of each two rows
    arr_zeros = arr_not1.mm(arr_not2.T)  # count the positions of both zeros of each two rows
    h_distance = m * torch.ones((n1, n2), device=device) - arr_zeros - arr_ones

    if isinstance(arr, list):
        return h_distance.mean()
    else:
        return h_distance.sum() / ((n1 * n2 - len(h_distance.diagonal())) + 1e-6)
    #   return h_distance[np.triu_indices(n1, k=1)]


def get_hammingdis(p=1, m=1):
    if p == 1:
        ac = lambda x: 1 / (1 + torch.exp(-m * x))
        norm = lambda x: x**2
    else:
        ac = norm = lambda x: x

    def hammingDistance_classwise(sigs, labels):
        if labels.device.type == 'cuda':
            labels = labels.cpu()
        sigs = ac(sigs)
        num_classes = labels.max()
        hreg_same_class = hreg_diff_class = 0
        if num_classes == 1:
            hreg_same_class = F.pdist(sigs.float(), p=p).mean()
        else:
            for i in range(num_classes.item()):
                class1 = norm(sigs[np.where(labels == i)].float())
                class_rest = norm(sigs[np.where(labels > i)].float())
                if len(class1) > 1:
                    hreg_same_class += F.pdist(class1, p=p).mean()
                if len(class1) > 0 and len(class_rest) > 0:
                    hreg_diff_class += torch.cdist(class1, class_rest, p=p).mean()
            class_last = norm(sigs[np.where(labels == num_classes)].float())
            hreg_same_class += F.pdist(class_last, p=p).mean()

        return hreg_same_class, hreg_diff_class

    return hammingDistance_classwise
