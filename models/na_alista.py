"""
file: models/na_alista.py
author: Xiaohan Chen
last modified: 2021.05.28

Implementation NA_ALISTA with support selection, transplanted from the official
GitHub repo.
"""

import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from .utils import shrink, shrink_ss


class NA_ALISTA(nn.Module):
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

        self.regressor = NormLSTMCB(dim=128)

        if opts.get('ss', True):
            self.ss = True
            maxp = opts.get('maxp', 13)
            p_per_layer = opts.get('p_per_layer', 1.2)
            p_schedule = np.array([(l+1) * p_per_layer for l in range(self.layers)])
            self.p_schedule = np.clip(p_schedule, 0.0, maxp) / 100.
        else:
            self.ss = False


    """
    Function: get_optimizer
    Purpose : Return the desired optimizer for the model.
    """
    def get_optimizer(
        self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3
    ):
        param_groups = []

        # Current layer
        param_groups.append(
            {
                'params': [self.gamma[layer-1], self.theta[layer-1]],
                'lr'    : init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': [self.gamma[i], self.theta[i]],
                        'lr'    : init_lr * (lr_decay_layer ** (layer-i-1))
                    }
                )

            # Stage decay
            stage_decay = lr_decay_stage2 if stage == 2 else lr_decay_stage3
            for group in param_groups:
                group['lr'] *= stage_decay

        return optim.Adam(param_groups)


    """
    Function: name
    Purpose : Identify the name of the model
    """
    def name(self):
        return 'NA_ALISTA'


    """
    Function: param_groups
    Purpose : Return the param_groups by layers
    """
    def param_groups(self):
        param_groups = []
        for gamma, theta in zip(self.gamma, self.theta):
            param_groups.append({'params': [gamma, theta]})
        return param_groups


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
        assert index >= 0

        if self.symm:
            A = self.W_gram
            Wt = self.Wt_gram
        else:
            A = self.A
            Wt = self.Wt

        # r = F.linear(x, self.A) - d
        # z = x - self.gamma[index] * F.linear(r, self.Wt)  
        r = F.linear(x, A) - d
        z = x - self.gamma[index] * F.linear(r, Wt)  

        if self.ss:
            Tx = shrink_ss(z, self.theta[index], self.p_schedule[index])
        else:
            Tx = shrink(z, self.theta[index])

        return Tx



    """
    Function: S_d(x)
    Purpose : This is used in LSKM for evaluating inferences.
    Notes   : There is the optional ability to include values for 'L' and 'tau',
              e.g., the same ones as used by ISTA.
    """
    def S(self, x, d, **kwargs):
        L   = kwargs.get('L',   self.L_ref)
        tau = kwargs.get('tau', self.tau)
        return x.sub(self.T(x,d,L=L,tau=tau))


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

        # Initialize the iterate xk.
        # Note the first dimension of xk is the batch size.
        # The second dimension is the size of each x^k, and the final is unity
        #   since x^k is a vector.
        xk = d.new_zeros(d.shape[0], self.n)

        cellstate, hidden = self.regressor.get_initial(d.shape[0])

        for i in range(K):
            a = F.linear(xk, self.A)
            b = a - d
            c = F.linear(b, self.Wt)
            pred, hidden, cellstate = self.regressor(b, c, hidden, cellstate)
            gamma = pred[:, :1]
            theta = pred[:, 1:]

            z = xk - gamma * c
            xk = shrink_ss(z, theta, self.p_schedule[i])

        return xk


class NormLSTMCB(nn.Module):
    def __init__(self, dim=128):
        super(NormLSTMCB, self).__init__()
        self.dim = dim
        self.lstm = nn.LSTMCell(2, dim)
        self.lll = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, 2)
        self.softplus = nn.Softplus()

        self.hidden = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))
        self.cellstate = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))

        self.bl1_mean = 0
        self.cl1_mean = 0
        self.bl1_std = 0
        self.cl1_std = 0
        self.initialized_normalizers = False

    def get_initial(self, batch_size):
        return (
            self.cellstate.unsqueeze(0).repeat(batch_size, 1),
            self.hidden.unsqueeze(0).repeat(batch_size, 1),
        )

    def forward(self, b, c, hidden, cellstate):
        bl1 = torch.norm(b, dim=1, p=1)
        cl1 = torch.norm(c, dim=1, p=1)
        if not self.initialized_normalizers:
            self.bl1_mean = bl1.mean().item()
            self.bl1_std = bl1.std().item()
            self.cl1_mean = cl1.mean().item()
            self.cl1_std = cl1.std().item()
            self.initialized_normalizers = True

        stack = torch.stack([(bl1 - self.bl1_mean) / self.bl1_std, (cl1 - self.cl1_mean) / self.cl1_std], dim=1)

        hidden, cellstate = self.lstm(stack, (hidden, cellstate))
        out = self.softplus(self.linear(torch.relu(self.lll(cellstate))))
        return out, hidden, cellstate


def test():
    return True


if __name__ == "__main__":
    test()

