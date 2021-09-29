'''
===================================================================================
Description: Models used for kernel linerization of koopman operator
Project: Data driven control for the multiagent fleet
Author: Wenjian Hao, Purdue University

Version: Sep / 2021
===================================================================================
'''

import torch
import torch.nn as nn

#========================================
# Build the lifting network
#========================================
class LNN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LNN, self).__init__()
        # hidden layers
        self.inputsize = dim_input
        self.hidd1size = 128
        self.outputsize = dim_output
        self.psi_hidden_layers = nn.Sequential(

            nn.Linear(self.inputsize, self.hidd1size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidd1size, self.outputsize, bias=True)
        )
        
    def forward(self, input):  
        input = input
        output = self.psi_hidden_layers(input)
        return output

#========================================
# Build the autodecoder network, if needed
#========================================
class DECODER_DNN(nn.Module):
    def __init__(self, dim_input, dim_output, dim_control):
        super(DECODER_DNN, self).__init__()
        # hidden layers
        self.inputsize = dim_input + dim_output + dim_control
        # self.inputsize = dim_output + dim_control
        self.hidd1size = 128
        self.hidd2size = 128
        self.hidd3size = 64
        self.outputsize = dim_output
        self.decoder_hidden_layers = nn.Sequential(
            
            nn.Linear(self.inputsize, self.hidd1size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidd1size, self.hidd2size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidd2size, self.hidd3size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidd3size, self.outputsize, bias=True)
        )
        
    def forward(self, input):  
        input = input
        output = self.decoder_hidden_layers(input)
        return output
