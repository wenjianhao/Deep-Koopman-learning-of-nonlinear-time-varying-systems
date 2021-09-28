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
from sklearn.externals import joblib
from functions import load_data

def process_data(data_name, NUM_PHISICAL_STATES):
    filename_in = data_name
    X, Y = load_data(filename_in)

    samples = X[0:(NUM_PHISICAL_STATES), :]
    label = Y[0:(NUM_PHISICAL_STATES), :]
    U_control = X[3, :]
    U_control = U_control.reshape(1,np.size(U_control))
    return samples, label, U_control

# load models
load_name = 'data/6_6.pth'
checkpoint = torch.load(load_name)
model_PSI = PSI_NN()
model_PSI.load_state_dict(checkpoint['model_PSI'])

NUM_PHISICAL_STATES = 3
data0 = 'data/data.pkl'
x0, y0, control0 = process_data(data0, NUM_PHISICAL_STATES)
koopman = np.matrix(y0) * np.linalg.pinv(np.matrix(x0))

NN_loss = 0
for j in range (len(x0[0])):
    x, y, u= x0, y0, control0
    trainingset = x[:, j]
    labelset = y[:, j]
    ulabelset = u[:,j]

    trset = torch.from_numpy(trainingset).float()
    labset = torch.from_numpy(labelset).float()
    ulabset = torch.from_numpy(ulabelset).float()
    
    psi_0 = np.matrix([1,0,0])
    psi_0 = torch.from_numpy(psi_0).float()
    psi_0 = model_PSI(psi_0)

    x_lift = model_PSI(trset) - psi_0
    x_next_lift = model_PSI(labset) - psi_0

    loss = torch.norm((x_next_lift - (x_next_lift * torch.pinverse(x_lift)) * x_lift), p=2)
    NN_loss += loss

print('average loss is: ', NN_loss.cpu().detach().numpy()/len(x0[0]))

linear_loss = 0
for i in range (len(x0[0])):
    x, y, u= x0, y0, control0
    trainingset = x[:, i]
    labelset = y[:, i]
    loss = np.linalg.norm((labelset.reshape(3,1) - koopman*trainingset.reshape(3,1)))
    linear_loss += loss
print('average loss is: ', linear_loss/len(x0[0]))