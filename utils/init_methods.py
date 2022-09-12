import math
import numpy as np
import torch


def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def normal_custom(w):
    fan_out, fan_in = w.shape[:2]
    std = np.sqrt(float(fan_in + fan_out) / float(fan_in * fan_out))                                                      
    return _no_grad_normal_(w, 0., std)

def uniform_custom(w):
    fan_out, fan_in = w.shape
    bound = np.abs(fan_in + fan_out) / float(fan_in * fan_out)
    torch.nn.init.uniform_(w, -bound, bound)

def he_uniform(w):
    torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))