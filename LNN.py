'''
Description: Paper codes of 'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." 
                            Automatica 159 (2024): 111372.'

Author: Wenjian Hao, AAE, Purdue University.

This is a file used to define the DNNs basis structure.

Start at: Sep 2021.
Last Revision: Jan 2022.
'''

import torch

import numpy as np
import torch.nn as nn

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def fanin_(size):
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

rbf = False
# rbf = True
# #========================================
# # Build the lifting network
# #========================================
if not rbf:
    class LNN(nn.Module):
        def __init__(self, dim_input, dim_output):
            super(LNN, self).__init__()
            # hidden layers
            self.inputsize = dim_input
            self.hidd1size = 16
            self.hidd2size = 256
            self.outputsize = dim_output
            self.psi_hidden_layers = nn.Sequential(

                nn.Linear(self.inputsize, self.hidd1size, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidd1size, self.hidd2size, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidd2size, self.outputsize, bias=True)
                
            )
            
        def forward(self, input, Test=False):  
            input = input
            output = self.psi_hidden_layers(input)
            return output

if rbf:
    ##network define
    class LNN(nn.Module):
        def __init__(self, dim_input, dim_output, init_w=3e-3):
            super(LNN, self).__init__()
            h1 = 16
            h2 = 256

            self.centres = nn.Parameter(torch.Tensor(1, dim_output))
            self.log_sigmas = nn.Parameter(torch.Tensor(dim_output))
            self.basis_func = gaussian
            self.reset_parameters()

            self.linear1 = nn.Linear(dim_input, h1)
            self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
                    
            self.linear2 = nn.Linear(h1, h2)
            self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                    
            self.linear3 = nn.Linear(h2, dim_output)
            self.linear3.weight.data.uniform_(-init_w, init_w)

            self.relu = nn.ReLU()

        def reset_parameters(self):
            nn.init.normal_(self.centres, 0, 1)
            nn.init.constant_(self.log_sigmas, 0)
            
        def forward(self, state, Test = False):
            x = self.linear1(state)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = self.linear3(x)

            c = self.centres.repeat(x.size(0),1)
            if Test:
                c = self.centres

            distances = (x - c).pow(2).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
            x = self.basis_func(distances)

            return x

# ohter RBFs
def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
