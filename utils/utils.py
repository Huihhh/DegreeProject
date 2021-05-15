import torch
import torch.utils.data as Data
from torch.utils.data.dataloader import DataLoader
import math


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
    feature_dataloader = DataLoader(feature_dataset,
                                        batch_size=64,
                                        num_workers=4,
                                        drop_last=False,
                                        pin_memory=True
                                        )
    return feature_dataloader