'''
===================================================================================
Description: Models used for kernel linerization of koopman operator
Project: Deep Koopman Representation for Time-varying Systems
Author: Wenjian Hao, Purdue University

Version: Sep 2021

Training data format, data.pkl:

 | x11 x12 ... x1t |
 | x21 x12 ... x1t |
 | ... ... ... ... |
 | xn1 xn2 ... xnt |
 | u11 u12 ... u1t |
 | ... ... ... ... |
 | um1 um2 ... umt |

 n: dimension of the states
 m: dimension of the control
 t: time series 
             
'''

#=====================
# Load third packages
#=====================
import os
import argparse
from utils import DKRC_training


#=====================
# Action
#=====================
train = True                            # Decide to start training or not
keep_train = True   #   False           # Keep training based on the previous model
plot_NN = False                         # Plot the structure of the lifting network and the decoder network

#=====================
# Set parameters
#=====================
if not os.path.exists("./SavedResults"):
    os.makedirs("./SavedResults")

results_path = 'SavedResults/'
data_name = 'data_collecting/data/trainingdata.pkl'
model_saved_name = 'liftnetwork.pth'
parser = argparse.ArgumentParser()
parser.add_argument("--dim_states", default=4, type=int)            # dimension of system states
parser.add_argument("--dim_lifting", default=16, type=int)          # dimension of the lifting
parser.add_argument("--dim_control", default=4, type=int)           # dimension of the control
parser.add_argument("--training_epoch", default=600, type=int)      # training epoch
parser.add_argument("--learning_rate", default=1e-5)                # learning rate of the optimizer
parser.add_argument("--decay_rate", default=1e-5)                   # training decay rate of the optimizer
args = parser.parse_args()

#=====================
# Training
#=====================
if train:
  DKRC = DKRC_training(results_path, data_name, model_saved_name, 
                      args.training_epoch, args.dim_states, 
                      args.dim_lifting, args.dim_control,
                      args.learning_rate, args.decay_rate, keep_train, plot_NN)
  DKRC.train_model()
