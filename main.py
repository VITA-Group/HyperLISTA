"""
file: main.py
author: Xiaohan Chen
last modified: 2021.10.06

Main script to perform the backpropagation based training for sparse coding task.
"""

import os
import numpy as np
import configargparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
import models
from data.sc import create_sc_data

# Argument Parsing
parser = configargparse.get_arg_parser(description='Configurations for ALISTA experiement')

parser.add('--model-name', type=str, metavar='STR',
           help='What model to use for the current experiment')
parser.add('--mode', type=str, default='train', metavar='STR',
           help='Train mode or test mode')

# Problem parameters
parser.add('--tau', type=float, default=0.1, metavar='FLOAT',
           help='Parameter for reg. term in the objective function')
parser.add('--m', type=int, default=250, metavar='INT',
           help='Number of rows in matrix A')
parser.add('--n', type=int, default=500, metavar='INT',
           help='Number of cols in matrix A')

# Model parameters
parser.add('--layers', type=int, default=16, metavar='INT',
           help='Number of layers of the neural network')
parser.add('--symm', action='store_true',
           help='Use the new symmetric matrix parameterization')

# Support selection parameters
parser.add('--no-ss', dest='ss', action='store_false',
           help='Control whether to use support selection or not')
parser.add('--ss-p-per-layer', type=float, default=1.2, metavar='FLOAT',
           help='Percentage of additional coordinates that bypass thresholding')
parser.add('--ss-maxp', type=float, default=13, metavar='FLOAT',
           help='Maximal percentage of coordinates in support selection')
parser.add('--ss-test-scale', type=float, default=None, metavar='FLOAT',
           help='Scale support selection percentage during testing')

# Parameters that control the approximations in HyperLISTA
parser.add('--approx', action='store_true',
           help='Do approximation to l1 error (and support [TBD])')
parser.add('--approx-support', action='store_true',
           help='Do approximation to the support size')
parser.add('--mag-ratio', type=float, default=0.0,
           help='Ratio to filter the approximated sparse vectors using maximum magnitude')
parser.add('--approx-ss', action='store_true',
           help='Do approximation to the support selection ratio')
parser.add('--no-learn-step-size', action='store_true',
           help='Learn step size parameters')

# Data parameters
parser.add('--x-dist', type=str, default='normal', metavar='STR',
           help='Distribution of non-zero elements in the sparse vectors')
parser.add('--x-mu', type=float, default=0.0, metavar='FLOAT',
           help='For training: mean of the Gaussian distribution of nonzero entries')
parser.add('--x-sigma', type=float, default=1.0, metavar='FLOAT',
           help='For training: std dev. of the Gaussian distribution of nonzero entries')
parser.add('--x-p', type=str, default='0.1', metavar='STR',
           help='For training: Bernoulli probability of entry being nonzero')
parser.add('--snr', type=float, default=1e10, metavar='FLOAT',
           help='SNR for additive Gaussian noise')
parser.add('--data-seed', type=int, default=118, metavar='INT',
           help='Random seed for the data generation')

# Training parameters
parser.add('--objective', type=str, default='GT', metavar='{OBJECTIVE,L2,L1,GT}',
           help='Objective used for the training')
parser.add('--save-dir', type=str, default='temp',
           help='Saving directory for saved models and logs')
parser.add('--train-size', type=int, default=51200, metavar='N',
           help='Number of training samples')
parser.add('--val-size', type=int, default=2048, metavar='N',
           help='Number of validation samples')
parser.add('--test-size', type=int, default=2048, metavar='N',
           help='Number of testing samples')
parser.add('--train-batch-size', type=int, default=512, metavar='N',
           help='Batch size for training')
parser.add('--val-batch-size', type=int, default=2048, metavar='N',
           help='Batch size for validation')
parser.add('--test-batch-size', type=int, default=2048, metavar='N',
           help='Batch size for testing')
parser.add('--init-lr', type=float, default=0.1, metavar='FLOAT',
           help='Initial learning rate')
parser.add('--lr-decay-layer', type=float, default=0.3, metavar='FLOAT',
           help='Decay learning rates of trained layers')
parser.add('--lr-decay-stage2', type=float, default=0.2,
           metavar='FLOAT', help='Decay rate for training stage2 in each layer')
parser.add('--lr-decay-stage3', type=float, default=0.02, metavar='FLOAT',
           help='Decay rate for training stage3 in each layer')
parser.add('--best-wait', type=int, default=5, metavar='N',
           help='Wait time for better validation performance')

opts, _ = parser.parse_known_args()

opts.learn_step_size = not opts.no_learn_step_size

# Save directory
opts.save_dir = os.path.join('results', opts.save_dir)
if not os.path.isdir(opts.save_dir):
    os.makedirs(opts.save_dir)
# Logging file
logger_file = os.path.join(opts.save_dir, 'train.log')
opts.logger = utils.setup_logger(logger_file)
opts.logger('Checkpoints will be saved to directory `{}`'.format(opts.save_dir))
opts.logger('Log file for training will be saved to file `{}`'.format(logger_file))

# Use cuda if it is available
opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
opts.dtype  = torch.float
opts.logger('Using device: {}'.format(opts.device)) # Output the type of device used
opts.logger('Using tau: {}'.format(opts.tau)) # Output the tau used in current exp


# -----------------------------------------------------------------------
#              Create data for training and validation
# -----------------------------------------------------------------------
train_seen_loader, val_seen_loader, test_seen_loader, A, W, W_gram, G = create_sc_data(opts)
A_TEN = torch.from_numpy(A).to(device=opts.device, dtype=opts.dtype)


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#                     LSKM Training Functions
#------------------------------------------------------------------------
#------------------------------------------------------------------------

"""
Function : make_train_step
Purpose  : This function defines a new function that is used to execute a single
           training step. The rationale for this is that the outer function
           enables one to choose the model, loss function, and optimizer. Once
           that is chosen, one can simply call 'train_step' to perform each
           training step with batches of the form (x_train, y_train).
"""

def make_train_step(model, optimizer):

    def train_step(d, network_layer, x_gt=None):
        model.train()             # Set the model to training mode
        x_pred = model(d, K=network_layer, x_gt=x_gt)
        loss = objective_val(x_pred, d, x_gt)
        loss.backward()
        optimizer.step() # Update the weights using the optimizer
        optimizer.zero_grad() # Set gradient to zero
        return loss.item()

    return train_step


"""
Function : objective_val
Purpose  : Compute the mean loss function estimate over a given batch of x and d.
Notes    : We provide four different choices of objective functions.
"""
def objective_val(x, d, x_gt=None, objective=None):

    if objective is None:
        objective = opts.objective

    #----------------------------------------------------------
    # Option 1: f(x) = (1/2)*|A*x-d|_2^2 + tau * |x|_1
    #           i.e. the LASSO objective function
    #----------------------------------------------------------
    if objective == 'OBJECTIVE':
        residual = ((F.linear(x, A_TEN) - d) ** 2.0).sum(dim=1).mean()
        l1_norm  = x.abs().sum(dim=1).mean()
        val      = residual / 2.0 + opts.tau * l1_norm

    #--------------------------------------------
    # Option 2:  (1/2) * | S(x^k) |_2^2
    #--------------------------------------------
    # Compute S(x) and take the average L2 norm over the whole (mini)-batch
    elif objective == 'L2':
        Sx  = model.S(x, d)
        val = (Sx**2.0).sum(dim=1).mean() / 2.0

    #--------------------------------------------
    # Option 3:  | S(x^k) |_1
    #--------------------------------------------
    elif objective == 'L1':
    # Compute S(x) and take the average L1 norm over the whole (mini)-batch
        Sx  = model.S(x, d)
        val = Sx.abs().sum(dim=1).mean()

    elif objective == 'GT':
        val = ((x - x_gt)**2).sum(dim=1).mean()

    elif objective == 'NMSE':
        l2 = ((x - x_gt)**2).mean()
        denom = (x_gt ** 2).mean()
        val = 10.0 * torch.log10(l2 / denom)

    else:
        raise ValueError('Invalid objective option {}'.format(opts.objective))

    return val


"""
Conduct Training
Note: Training occurs layer-wise first. Each the parameters of each layer are
      learned successively, with all the other variables held fixed. That is,
      theta[0] and gamma[0] are learned first, with the cost function being
                        \E [ objective_val(x^0) ].
      Then theta[1] and gamma[1] are learned, holding the variables already
      trained fixed, but with x^2 in place of x^1 in the cost function. This is
      done for all of the layers. After this takes place, a final stage of
      training occurs where all of the variables are included in the
      optimization (i.e., the entire network is used for gradient computations).
"""

torch.manual_seed(42) # set a seed for reproducability
if opts.model_name == 'LISTA':
    model = models.LISTA(A, W, None, None, opts.layers, opts.tau)
elif opts.model_name == 'ALISTA':
    model = models.ALISTA(A, W, W_gram, G, opts.layers, opts.tau,
                          ss=opts.ss,
                          p_per_layer=opts.ss_p_per_layer,
                          maxp=opts.ss_maxp,
                          # learn_step_size=opts.learn_step_size,
                          symm=opts.symm)
elif opts.model_name == 'AdaLISTA':
    model = models.AdaLISTA(A, W, opts.layers, opts.tau, momentum=False)
    opts.ss = False
elif opts.model_name == 'AdaLFISTA':
    model = models.AdaLISTA(A, W, opts.layers, opts.tau, momentum=True)
    opts.ss = False
elif opts.model_name == 'MMALISTAUntied':
    model = models.MMALISTAUntied(A, W, W_gram, G, opts.layers, opts.tau,
                                  ss=opts.ss,
                                  p_per_layer=opts.ss_p_per_layer,
                                  maxp=opts.ss_maxp,
                                  learn_step_size=opts.learn_step_size,
                                  symm=opts.symm)
elif opts.model_name == 'AdaptiveMMALISTA':
    model = models.AdaptiveMMALISTA(A, W, W_gram, G, opts.layers, opts.tau,
                                    ss=opts.ss,
                                    p_per_layer=opts.ss_p_per_layer,
                                    maxp=opts.ss_maxp,
                                    learn_step_size=opts.learn_step_size,
                                    approx=opts.approx,
                                    approx_support=opts.approx_support,
                                    mag_ratio=opts.mag_ratio,
                                    approx_ss=opts.approx_ss,
                                    symm=opts.symm)
else:
    raise ValueError('Invalid model name')

model = model.to(device=opts.device, dtype=opts.dtype) # move the model to current device

if opts.mode == 'train':
    training_losses   = [] # initialize the array storing training loss function
    validation_losses = [] # initialize the array storing validation loss function

    # Conduct training layer-wise in increasing depth.
    for j in range(opts.layers):
        current_layer = j + 1

        epoch = 0
        batch_losses = [] # Initialize batch losses
        # Loop over stage 1,2,3
        for stage in range(1, 4):
            # Set up optimizer
            optimizer = model.get_optimizer(
                layer = current_layer,
                stage = stage,
                init_lr = opts.init_lr,
                lr_decay_layer  = opts.lr_decay_layer,
                lr_decay_stage2 = opts.lr_decay_stage2,
                lr_decay_stage3 = opts.lr_decay_stage3,
            )
            best_val_nmse = 1e30
            best_val_epoch = epoch  # Starting each stage, the best epoch is the current epoch
            opts.logger('Training layer {} - stage {}'.format(current_layer, stage))
            # print(optimizer)

            # Define the function for updating model
            train_step = make_train_step(model, optimizer)

            while True:
                for x_batch, d_batch in train_seen_loader: # Loop over all training batches
                    x_batch = x_batch.to(device=opts.device, dtype=opts.dtype)
                    d_batch = d_batch.to(device=opts.device, dtype=opts.dtype)
                    loss = train_step(d_batch, network_layer=current_layer, x_gt=x_batch)
                    batch_losses.append(loss) # Add loss to list

                training_loss = np.mean(batch_losses) # Compute the average of the batch losses
                training_losses.append(training_loss) # Append this new value to the array of losses
                epoch += 1

                # Do validation
                with torch.no_grad(): # Turn off gradient usage when using model
                    val_losses = [] # Initialize list of validation losses
                    for x_val, d_val in val_seen_loader:  # Loop over all validation batches
                        model.eval() # Set the model to evaluation mode
                        d_val    = d_val.to(device=opts.device, dtype=opts.dtype)
                        x_val    = x_val.to(device=opts.device, dtype=opts.dtype)
                        x_pred   = model(d_val, K=current_layer, x_gt=x_val) # Infer the value of x from d
                        val_loss = objective_val(x_pred, d_val, x_val, 'NMSE').item()
                        val_losses.append(val_loss)  # Add current loss to list
                    validation_loss = np.mean(val_losses) # Compute the average of the batch losses
                    validation_losses.append(validation_loss) # Append this new value to the array of losses

                    # output the epoch results to the terminal
                    opts.logger(
                        '[%(first)d] Training loss: %(second).5e\t Validation loss: %(third)0.5e' % \
                        {"first":epoch, "second":training_loss, "third":validation_loss}
                    )

                if validation_loss < best_val_nmse:
                    best_val_nmse = validation_loss
                    best_val_epoch = epoch
                if epoch - best_val_epoch > opts.best_wait:
                    break

        checkpoint_name = model.name() + '.pt'
        save_path = os.path.join(opts.save_dir, checkpoint_name)
        opts.logger(model.state_dict())
        torch.save(model.state_dict(), save_path)
        opts.logger('Saved the model to file: ' + save_path)

elif opts.mode == 'test':
    checkpoint_name = model.name() + '.pt'
    save_path = os.path.join(opts.save_dir, checkpoint_name)
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    if opts.ss:
        if opts.ss_test_scale is not None:
            model.p_schedule *= opts.ss_test_scale
        if not opts.approx_ss:
            print(model.p_schedule)
    else:
        print('support selection is not used')

    testing_losses = [0.0]
    for current_layer in range(1, model.layers + 1):
        # Do testing
        with torch.no_grad(): # Turn off gradient usage when using model
            test_losses = [] # Initialize list of testing losses
            for x_test, d_test in test_seen_loader:  # Loop over all test batches
                model.eval() # Set the model to evaluation mode
                d_test   = d_test.to(device=opts.device, dtype=opts.dtype)
                x_test   = x_test.to(device=opts.device, dtype=opts.dtype)
                x_pred   = model(d_test, K=current_layer, x_gt=x_test) # Infer the value of x from d
                test_loss = objective_val(x_pred, d_test, x_test, 'NMSE').item()
                test_losses.append(test_loss)  # Add current loss to list
            testing_loss = np.mean(test_losses) # Compute the average of the batch losses
            testing_losses.append(testing_loss) # Append this new value to the array of losses

    # output the epoch results to the terminal
    opts.logger('Testing losses:')
    for t_loss in testing_losses:
        opts.logger('{}'.format(t_loss))

