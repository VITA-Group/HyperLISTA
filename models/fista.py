"""
file: models/lista.py
author: Xiaohan Chen
last modified: 2021.05.28

Implementation LISTA with support selection.
"""

import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from .utils import shrink, shrink_ss


class FISTA(nn.Module):
    def __init__(self, A, layers, tau, **opts):
        super().__init__()

        self.register_buffer('A', torch.from_numpy(A))
        self.register_buffer('At', torch.from_numpy(np.transpose(A)))

        self.m, self.n = self.A.size()

        self.layers = layers # Number of layers in the network
        self.tau    = tau    # Parameter for problem definition
        # Compute the norm |A^t*A|_2 and assign this to L
        self.L_np   = np.linalg.norm(np.matmul(A.transpose(),A), ord=2)
        self.register_buffer('L_ref', self.L_np * torch.ones(1,1))
        # print(self.L_ref)
        self.register_buffer('gamma', 1 / self.L_ref)


    def name(self):
        return 'FISTA'


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

        r = F.linear(x, self.A) - d
        z = x - self.gamma * F.linear(r, self.At)  
        
        # print(tau)
        Tx = shrink(z, self.gamma * tau)

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
        # print(self.tau)

        with torch.no_grad():
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
                t_next = 0.5 + math.sqrt(1.0 + 4.0 * tk**2) / 2.0
                z_next = x_next + (tk - 1.0) / t_next * (x_next - xk)
                # print((tk -1.0)/t_next)

                # Process iteration
                xk = x_next
                zk = z_next
                tk = t_next

        return xk


def test():
    return True


if __name__ == "__main__":
    test()


