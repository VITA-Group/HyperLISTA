"""
file: models/adaptive_mm_alista.py
author: Xiaohan Chen
last modified: 2021.05.02

Implementation ALISTA with single parameter and support selection, with momentum.
"""

import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from .utils import shrink, shrink_ss


class AdaptiveMMALISTA(nn.Module):
    def __init__(self, A, W, W_gram, G, layers, tau, **opts):
        super().__init__()

        self.register_buffer('A', torch.from_numpy(A))
        self.register_buffer('At', torch.from_numpy(np.transpose(A)))

        self.m, self.n = self.A.size()

        self.register_buffer('W', torch.from_numpy(W))
        self.register_buffer('Wt', torch.from_numpy(np.transpose(W)))

        if W_gram is not None:
            self.register_buffer('W_gram', torch.from_numpy(W_gram))
            self.register_buffer('Wt_gram', torch.from_numpy(np.transpose(W_gram)))
        else:
            self.W_gram = self.Wt_gram = None
        if G is not None:
            self.register_buffer('G', torch.from_numpy(G))
        else:
            self.G = None

        self.layers = layers # Number of layers in the network
        self.tau    = tau    # Parameter for problem definition
        self.symm   = opts.get('symm', False)
        # Compute the norm |A^t*A|_2 and assign this to L
        self.L_np   = np.linalg.norm(np.matmul(A.transpose(),A), ord=2)
        self.register_buffer('L_ref', self.L_np * torch.ones(1,1))

        approx = opts.get('approx', False)
        approx_support = opts.get('approx_support', False)
        self.approx = approx
        self.approx_support = approx_support
        self.mag_ratio = opts.get('mag_ratio', 0.0)
        if self.approx:
            A_pinv = np.linalg.pinv(A)
            self.register_buffer('A_pinv', torch.from_numpy(A_pinv))

        if opts.get('ss', True):
            self.ss = True
            self.approx_ss = opts.get('approx_ss', False)
            if self.approx_ss:
                self.c_ss = nn.Parameter(1e-4 * torch.ones(1,1))
            else:
                maxp = opts.get('maxp', 13)
                p_per_layer = opts.get('p_per_layer', 1.2)
                p_schedule = np.array([(l+1) * p_per_layer for l in range(self.layers)])
                self.p_schedule = np.clip(p_schedule, 0.0, maxp) / 100.
        else:
            self.ss = False

        #-------------------------------------
        # Define the parameters for the model
        #-------------------------------------
        self.learn_step_size = opts.get('learn_step_size', False)
        if self.learn_step_size:
            self.gamma = nn.ParameterList()
            for i in range(self.layers):
                self.gamma.append(nn.Parameter(1.0 / self.L_ref))
        else:
            self.gamma = [0.05] * self.layers

        self.c_beta = nn.Parameter(1e-2 * torch.ones(1,1))
        self.c_theta = nn.Parameter(1e-6 * torch.ones(1,1))


    """
    Function: get_optimizer
    Purpose : Return the desired optimizer for the model.
    """
    def get_optimizer(
        self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3
    ):
        param_groups = []

        # Parameter group for c_beta
        c_beta_param_group = {
            'params': [self.c_beta],
            'lr'    : init_lr * (lr_decay_layer ** (layer - 1))
        }
        param_groups.append(c_beta_param_group)

        # Parameter group for c_theta
        c_theta_param_group = {
            'params': [self.c_theta],
            'lr'    : init_lr * (lr_decay_layer ** (layer - 1))
        }
        param_groups.append(c_theta_param_group)

        # Parameter group for c_ss
        if self.approx_ss:
            c_ss_param_group = {
                'params': [self.c_ss],
                'lr'    : init_lr * (lr_decay_layer ** (layer - 1))
            }
            param_groups.append(c_ss_param_group)

        # Current layer
        if self.learn_step_size:
            param_groups.append(
                {
                    'params': [self.gamma[layer-1]],
                    'lr'    : init_lr
                }
            )
            # Stage 2 / 3
            if stage > 1:
                # Previous layers
                for i in range(layer - 1):
                    param_groups.append(
                        {
                            'params': [self.gamma[i]],
                            'lr'    : init_lr * (lr_decay_layer ** (layer-i-1))
                        }
                    )

        # Stage decay
        if stage > 1:
            stage_decay = lr_decay_stage2 if stage == 2 else lr_decay_stage3
            for group in param_groups:
                group['lr'] *= stage_decay

        return optim.Adam(param_groups)


    """
    Function: param_groups
    Purpose : Return the param_groups by layers
    """
    def param_groups(self):
        # param_groups = []
        # for _ in range()
        #     param_groups.append({'params': [gamma, theta]})
        # return param_groups
        pass


    """
    Function: name
    Purpose : Identify the name of the model
    """
    def name(self):
        return 'Adaptive-MM-ALISTA'


    """
    Function: two_norm
    Purpose : Compute the Euclidean norms of the rows of the input tensor.
              The number of rows is the number of samples in a batch.
    """
    def two_norm(self, z):
        return (z ** 2).sum(dim=1).sqrt()


    """
    Function: T_d(x)
    Purpose : The operator T(x) is the core nonexpansive operator defining the
              KM iteration. Here T(x) is a Forward-Backward Splitting for the
              LASSO problem, i.e., the ISTA operator. Note we include an
              argument 'd' since the operator T varies, depending on the input
              data. Thus, in the paper we add a subscript 'd'.
    """
    def T(self, x, d, **kwargs):
        # Enable the ability to use particular parameters when T(x,d) is called
        #   as part of the loss function evaluation. For example, the choice of
        #   'L' can change.
        tau    = kwargs.get('tau', self.tau)
        index  = kwargs.get('index', -1)
        x_prev = kwargs.get('x_prev', None)
        assert index >= 0
        assert x_prev is not None

        x_gt   = kwargs.get('x_gt', None)
        if not self.approx or not self.approx_support:
            assert x_gt is not None

        if self.symm:
            A = self.W_gram
            Wt = self.Wt_gram
        else:
            A = self.A
            Wt = self.Wt

        r = F.linear(x, A) - d
        z = x - self.gamma[index] * F.linear(r, Wt)

        # Momentum term - get beta first
        if self.approx:
            if self.approx_support:
                x_abs = x.abs()
                max_mag = x_abs.max(dim=1, keepdim=True)[0]
                mag_threshold = max_mag * self.mag_ratio
                approx_sp = torch.sum((x_abs > mag_threshold).float(), dim=1, keepdim=True)
                beta = self.c_beta * approx_sp
            else:
                sp = torch.sum((x_gt != 0.0).float(), dim=1, keepdim=True)
                beta = self.c_beta * sp
        else:
            sp = torch.sum((x_gt != 0.0).float(), dim=1, keepdim=True)
            beta = self.c_beta * sp
        # Add momentum
        z += beta * (x - x_prev)

        # Get threshold using:  `theta = c * gamma * ||x - x_k||_1`
        if self.approx:
            pinv_res = F.linear(r, self.A_pinv)
            approx_l1_err = torch.norm(pinv_res, p=1, dim=1, keepdim=True)
            theta = self.c_theta * self.gamma[index] * approx_l1_err
        else:
            l1_err = torch.sum((x - x_gt).abs(), dim=1, keepdim=True)
            theta = self.c_theta * self.gamma[index] * l1_err

        if self.ss:
            # Get support selection ratio
            if self.approx and self.approx_ss:
                with torch.no_grad():
                    pinv_b = F.linear(d, self.A_pinv)
                    l1_pinv_b = torch.norm(pinv_b, p=1, dim=1, keepdim=True) + 1e-20  # Numerical stability
                    # p = F.relu(1.0 - self.c_ss * approx_l1_err / l1_pinv_b).reshape(-1)
                    temp = (l1_pinv_b / approx_l1_err).log()
                    p = torch.clamp(self.c_ss * temp, 0.0, 1.0).reshape(-1)
            else:
                p = self.p_schedule[index]
            Tx = shrink_ss(z, theta, p)
        else:
            p = torch.zeros(1,1)
            Tx = shrink(z, theta)  #, self.p_schedule[index])

        # print('approx_l1_err: {}  beta :{}  theta: {}  p: {}'.format(approx_l1_err.mean(), beta.mean(), theta.mean(), p.mean()))

        return Tx


    """
    Function: forward
    Purpose : This function defines the feed forward operation of the LSKM network.
    Notes   : Currently, the choice of v^k is according to the Bad Broyden updates.
              We use x^1 = 0. We also allow for the ability to optionally input
              a desired number of layers to use rather than the full network.
              This allows the network to be trained layer-wise and also is
              helpful for debugging.
              REVISION: Need to return and add an optional argument
              `compute_loss` that will return, in addition to x^k, loss values.
              This would make the code execute a fair bit faster and simpler
              while generating plots.
    """
    def forward(self, d, **kwargs):
        # Optional to input the desired number of layers. The default value is
        #   the number of layers.
        K = kwargs.get('K', self.layers)
        # Ensure  K <= self.layers
        K = min(K, self.layers)

        x_gt = kwargs.get('x_gt', None)
        if not self.approx or not self.approx_support:
            assert x_gt is not None

        if self.symm:
            d = F.linear(d, self.G)

        # Initialize the iterate xk.
        # Note the first dimension of xk is the batch size.
        # The second dimension is the size of each x^k, and the final is unity
        #   since x^k is a vector.
        xk = d.new_zeros(d.shape[0], self.n)
        x_prev = d.new_zeros(d.shape[0], self.n)

        for i in range(K):
            x_next = self.T(xk, d, index=i, x_gt=x_gt, x_prev=x_prev)
            x_prev = xk
            xk = x_next

        return xk


def test():
    return True


if __name__ == "__main__":
    test()


