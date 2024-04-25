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

# DNN basis function for Koopman operator
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

# For comparison
class LNNsl(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LNNsl, self).__init__()
        # hidden layers
        self.inputsize = dim_input
        self.hidd1size = 16
        self.hidd2size = 32
        self.outputsize = dim_output
        self.psi_hidden_layers = nn.Sequential(

            nn.Linear(self.inputsize, self.hidd1size, bias=True),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(self.hidd1size, self.hidd2size, bias=True),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(self.hidd2size, self.outputsize, bias=True)
        )
        
    def forward(self, input):  
        input = input
        output = self.psi_hidden_layers(input)
        return output
