'''
29/ OCT / 2019 Clemson University

GOAL: DEEP EDMD TO APPROXIMATE APPROPRIATE PSI FUNCTION AND [A, B] MATRIX 

LOSS FUNCTION: 
              min { 2_norm(phi(Xt+1) - A * phi(Xt)) - B * U) + lambda1 * 2_norm(K) + lambda2 * 2_norm(B) + lambda3 * 1_norm(phi)}

C MATRIX:
        C = X * pseudoinverse(X_lifted)       
             
'''

import math
import numpy as np
import torch
import pickle
import scipy
import control

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import PSI_NN
import joblib
from functions import data_process, train_model, computer_a_b_c
# from Controller import MPC, FINITE_LQR

# DECISION
train = True
compute_A_B_C_matrix = True
LQR_controller = False
MPC_controller = False

##################################
# Loading and preprocessing data #
##################################
data_path = 'data/'
data_name = 'data/data.pkl'
NUM_PHISICAL_STATES = 3
NUM_LIFT_DIM = 8
NUM_CONTROL = 1
num_epoch = 70
train_terminal = 0.082
modelname = 'data/6_6.pth'
np.set_printoptions(precision=4, suppress=True, linewidth=120)
train_samples, train_label, train_u, valid_samples, valid_label, valid_u = data_process(data_name, NUM_PHISICAL_STATES)
# train_samples -= train_samples.mean()
# train_label -= train_label.mean()
# train_u -= train_u.mean()
# print(train_samples.mean())
# print(train_label.mean())
# print(train_u.mean())

###########################################
# Train psi function and get A,B,C matrix #
###########################################
if train:
  train_model(train_samples, train_label, train_u, valid_samples, valid_label, valid_u, NUM_PHISICAL_STATES, NUM_LIFT_DIM, NUM_CONTROL, data_path, modelname, num_epoch)

##########################
# compute A B C matrices #
##########################
if compute_A_B_C_matrix:
  computer_a_b_c(NUM_PHISICAL_STATES, NUM_LIFT_DIM)

##########################
# Deployment             #
##########################
if LQR_controller:
  run = FINITE_LQR()
if MPC_controller:
  run = MPC()
