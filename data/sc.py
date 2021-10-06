"""
File:    sc.py
Created: September 9, 2019
Revised: March 11, 2020
Author:  Howard Heaton, Xiaohan Chen
Purpose: Define a function for generating the data used in training
         and/or testing of the LSKM Model for the LASSO Problem.
"""

import os
import scipy.io
import numpy as np

import torch
from   torch.utils.data         import Dataset, TensorDataset, DataLoader
from   torch.utils.data.dataset import random_split


def create_sc_data(opts, shuffle_train=True):

    # Reproducibility
    # seed = opts.get('seed', 118)  # Manually fix random seed
    seed = getattr(opts, 'data_seed', 118)
    np.random.seed(seed)
    torch.manual_seed(seed+1)

    #-------------------------------------
    # Generate A = Gaussian matrix
    #-------------------------------------
    assert os.path.isfile('data/A_sc.mat')
    A = scipy.io.loadmat('data/A_sc.mat')['A']
    opts.logger('File `data/A_sc.mat` exists. Loaded dictionary A from it')
    m, n = A.shape

    #-------------------------------------
    # Load pre-computed weight matrix W
    #-------------------------------------
    assert os.path.isfile('data/W_sc.mat')
    W = scipy.io.loadmat('data/W_sc.mat')['W']
    opts.logger('Loaded matrix W from `data/W_sc.mat` file')

    #--------------------------------------------------------------------------
    # Load pre-computed W_gram and G matrices for symmetric parameterization
    #--------------------------------------------------------------------------
    if opts.symm:
        assert os.path.isfile('data/W_gram_sc.mat')
        assert os.path.isfile('data/G_sc.mat')
        W_gram = scipy.io.loadmat('data/W_gram_sc.mat')['W_gram']
        opts.logger('Loaded matrix W_gram from `data/W_gram_sc.mat` file')
        G = scipy.io.loadmat('data/G_sc.mat')['G']
        opts.logger('Loaded matrix G from `data/G_sc.mat` file')
    else:
        W_gram = G = None

    #-------------------------------------
    # Generate x_seen = Bernoulli * Gaussian
    #-------------------------------------
    # process the sparsity ratio x_p
    ps = list(map(float, opts.x_p.split('-')))
    if len(ps) == 1:
        ps *= 2
    elif len(ps) == 2:
        assert ps[0] <= ps[1]
    else:
        raise ValueError('invalid sparsity option --x-p for')

    seen_data_size  = opts.train_size + opts.val_size + opts.test_size
    p = np.random.uniform(low=ps[0], high=ps[1], size=[seen_data_size, opts.n])
    bernoulli_terms = np.random.binomial(size=[seen_data_size, opts.n], n=1, p=p)
    gaussian_terms  = np.random.normal(size=[seen_data_size, opts.n], loc=opts.x_mu, scale=opts.x_sigma)
    x               = np.multiply(bernoulli_terms, gaussian_terms)

    #-------------------------------------
    # Generate d = A*x + e
    #-------------------------------------
    d = np.matmul(A, x.transpose()) # Compute d = Ax
    # d = d + np.random.normal(0, 0.1/opts.m, [opts.m, seen_data_size]) # Add noise
    d = d.transpose() # Take the transpose

    if opts.snr and opts.snr < 100000:
        std = np.std(d, axis=1) * np.power (10.0, -opts.snr/20.0)
        noise = np.random.normal(size=d.shape, scale=std).astype(d.dtype)
        print('noise added')
    else:
        noise = 0

    d += noise

    # Transform x and d into tensors
    x_tensor = torch.from_numpy(x).float()
    d_tensor = torch.from_numpy(d).float()

    # Create a dataset from our training inputs 'x_tensor' and outputs 'y_tensor'
    seen_dataset = TensorDataset(x_tensor, d_tensor)

    # Split the seen dataset into training, validation and testing portions
    train_seen_dataset, val_seen_dataset, test_seen_dataset = random_split(
        seen_dataset, [opts.train_size, opts.val_size, opts.test_size])

    # Create loaders for batches from training and validation datasets
    train_seen_loader  = DataLoader(dataset=train_seen_dataset,
                                    batch_size=opts.train_batch_size,
                                    shuffle=shuffle_train)
    val_seen_loader    = DataLoader(dataset=val_seen_dataset,
                                    batch_size=opts.val_batch_size,
                                    shuffle=False)
    test_seen_loader   = DataLoader(dataset=test_seen_dataset,
                                    batch_size=opts.test_batch_size,
                                    shuffle=False)

    return train_seen_loader, val_seen_loader, test_seen_loader, A, W, W_gram, G

