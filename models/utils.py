import torch
import torch.nn as nn
import torch.nn.functional as F


def shrink(x, theta):
    return x.sign() * F.relu(x.abs() - theta)


def shrink_ss(x, theta, p):
    x_abs = x.abs()
    threshold = torch.quantile(x_abs, 1-p, dim=1, keepdims=True)
    if isinstance(p, torch.Tensor) and p.numel() > 1:
        threshold = torch.stack([threshold[i,i,0] for i in range(p.numel())]).unsqueeze(1)

    bypass = torch.logical_and(x_abs >= threshold, x_abs >= theta).detach()
    output = torch.where(bypass, x, shrink(x, theta))

    return output

