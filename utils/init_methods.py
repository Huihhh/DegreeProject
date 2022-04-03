import numpy as np
import torch


def range_uniform(x, a=0.8, b=1.2):
    n = x.shape[0] // 2
    x1 = torch.nn.init.uniform_(x[:n, ], a, b)
    x2 = torch.nn.init.uniform_(x[n:, ], -b, -a)
    return torch.cat([x1, x2], 0)


def tanh(w, a=-np.pi, b=np.pi):
    n, m = w.shape
    ww = np.zeros((n, m))
    for i in range(m):
        wn = np.tanh(np.linspace(a, b, n))
        ww[:, i] = wn
    w = torch.tensor(ww,
                     requires_grad=True,
                     device="cuda" if torch.cuda.is_available() else 'cpu')
    return w


def cos(w, a=-np.pi, b=np.pi):
    n, = w.shape
    wn = np.cos(np.linspace(a, b, n))
    w = torch.tensor(wn,
                     requires_grad=True,
                     device="cuda" if torch.cuda.is_available() else 'cpu')
    return w


def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def normal_custom(w):
    fan_out, fan_in = w.shape
    std = np.sqrt(1/2 * float(fan_in + fan_out) / float(fan_in * fan_out))
    return _no_grad_normal_(w, 0., std)

def normal_custom1(w):
    fan_out, fan_in = w.shape
    g = max(fan_in, fan_out) / min(fan_out, fan_in)
    std = np.sqrt(0.5**g * float(fan_in + fan_out) / (float(fan_in * fan_out)))
    return _no_grad_normal_(w, 0., std)

def uniform_custom(w):
    fan_out, fan_in = w.shape
    bound = np.abs(fan_in + fan_out) / float(fan_in * fan_out)
    torch.nn.init.uniform_(w, -bound, bound)