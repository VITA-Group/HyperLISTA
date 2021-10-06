"""
file: models/ada_lista.py
author: Xiaohan Chen
last modified: 2021.08.10

Implementation Ada-LISTA.
"""

import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from .utils import shrink, shrink_ss


class AdaLISTA(nn.Module):
    def __init__(self, A, W, layers, tau, **opts):
        super().__init__()

        self.register_buffer('A', torch.from_numpy(A))
        self.register_buffer('At', torch.from_numpy(np.transpose(A)))

        self.m, self.n = self.A.size()

        self.register_buffer('W', torch.from_numpy(W))
        self.register_buffer('Wt', torch.from_numpy(np.transpose(W)))

        self.layers = layers # Number of layers in the network
        self.tau    = tau    # Parameter for problem definition
        # Compute the norm |A^t*A|_2 and assign this to L
        self.L_np   = np.linalg.norm(np.matmul(A.T,A), ord=2)
        self.register_buffer('L_ref', self.L_np * torch.ones(1,1))

        # Check if using Ada-LFISTA
        self.momentum = opts.get('momentum', False)

        #-------------------------------------
        # Define the parameters for the model
        #-------------------------------------
        self.W1 = nn.Parameter(torch.eye(self.m))
        self.W2 = nn.Parameter(torch.eye(self.m))

        self.gamma   = nn.ParameterList()
        self.theta   = nn.ParameterList()
        for i in range(self.layers):
            self.gamma.append(nn.Parameter(1.0 / self.L_ref))
            self.theta.append(nn.Parameter(self.tau / self.L_ref))


    """
    Function: get_optimizer
    Purpose : Return the desired optimizer for the model.
    """
    def get_optimizer(
        self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3
    ):
        param_groups = []

        # W1, W2 group
        param_groups.append(
            {
                'params': [self.W1, self.W2],
                'lr'    : init_lr * (lr_decay_layer ** (layer - 1))
            }
        )

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
        return 'Ada-LISTA' if not self.momentum else 'Ada-LFISTA'


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

        mat1 = torch.chain_matmul(self.A.T, self.W1.T, self.W1, self.A)
        mat2 = torch.matmul(self.A.T, self.W2.T)

        hidden = x - self.gamma[index] * F.linear(x, mat1) + self.gamma[index] * F.linear(d, mat2)
        Tx = shrink(hidden, self.theta[index])

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
        zk = d.new_zeros(d.shape[0], self.n)
        tk = 1.0

        for i in range(K):
            x_next = self.T(zk, d, index=i)

            # Process momentum
            if self.momentum:
                t_next = 0.5 + math.sqrt(1.0 + 4.0 * tk**2) / 2.0
                z_next = x_next + (tk - 1.0) / t_next * (x_next - xk)
            else:
                t_next = 1.0
                z_next = x_next

            # Process iteration
            xk = x_next
            zk = z_next
            tk = t_next

        return xk


def test():
    return True


if __name__ == "__main__":
    test()


