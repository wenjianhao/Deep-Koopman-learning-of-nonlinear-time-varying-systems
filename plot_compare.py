'''
Description: Paper codes of 'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." 
                            Automatica 159 (2024): 111372.'

Author: Wenjian Hao, Purdue University.

This file is used for results visualization and comparison.

Start at: Sep 2021.
Last Revision: Jan 2022.
'''

# some third packages
import torch
import joblib

import numpy as np
import matplotlib.pyplot as plt

from LNN import LNN
from LNNsl import LNNsl
from numpy import linalg as LA

# load files and model
savedemo = 'SavedResults/dktvdemo.pkl'
load_name = 'SavedResults/nnbasis/'
model_saved_name = 'liftnetwork.pth'
fileastack = 'SavedResults/Astk.pkl'
filecstack = 'SavedResults/Cstk.pkl'
xfile = 'SavedResults/xdata.pkl'
tvdpred = 'SavedResults/tvdpred.pkl' # TVDMD results
Astack = joblib.load(fileastack)
Cstack = joblib.load(filecstack)
truestate = joblib.load(xfile)
tvdmdstore = joblib.load(tvdpred)

# plot labels and font sizes
labelsize = 15
dkstore = []
beta0 = 100
beta = 10
DKbasis = LNN(dim_input=2, dim_output=6)

# get the tvdmd results
for i in range(beta0, truestate.shape[1]):
    dynind = int(np.floor((i-beta0)/beta))   
    # load the nn basis
    indexb = int(dynind+1)
    basis = load_name + str(indexb) + model_saved_name
    checkpoint = torch.load(basis, map_location=torch.device('cpu'))
    DKbasis.load_state_dict(checkpoint['model_lifting'])
    xt = torch.from_numpy(np.array(truestate[:,i]).T).float()
    cur_state_lifted = DKbasis.forward(xt, Test=True).cpu().detach().numpy()
    dkpred = np.matrix(Cstack[:,:,indexb])*np.matrix(Astack[:,:,indexb])*np.matrix(cur_state_lifted).T
    dkstore.append(np.array(dkpred.T).squeeze())
dkpre = np.array(dkstore).T

# load files and model when \gamma=0.8
savedemo8 = 'SavedResults/SavedResults08/dktvdemo.pkl'
load_name8 = 'SavedResults/SavedResults08/nnbasis/'
fileastack8 = 'SavedResults/SavedResults08/Astk.pkl'
filecstack8 = 'SavedResults/SavedResults08/Cstk.pkl'
xfile8 = 'SavedResults/SavedResults08/xdata.pkl'
tvdpred8 = 'SavedResults/SavedResults08/tvdpred.pkl'
Astack8 = joblib.load(fileastack8)
Cstack8 = joblib.load(filecstack8)
truestate8 = joblib.load(xfile8)
tvdmdstore8 = joblib.load(tvdpred8)

dkstore8 = []
DKbasis8 = LNNsl(dim_input=2, dim_output=6)
for i in range(beta0, truestate8.shape[1]):
    dynind = int(np.floor((i-beta0)/beta))   
    # load the nn basis
    indexb = int(dynind+1)
    basis = load_name8 + str(indexb) + model_saved_name
    checkpoint = torch.load(basis, map_location=torch.device('cpu'))
    DKbasis8.load_state_dict(checkpoint['model_lifting'])
    cur_state_lifted = DKbasis8.forward(torch.from_numpy(np.array(truestate8[:,i]).T).float()).cpu().detach().numpy()
    dkpred8 = np.matrix(Cstack8[:,:,indexb])*np.matrix(Astack8[:,:,indexb])*np.matrix(cur_state_lifted).T
    dkstore8.append(np.array(dkpred8.T).squeeze())
dkpre8 = np.array(dkstore8).T

e18 = LA.norm((tvdmdstore8[:,:99]-truestate8[:, beta0+1:]), axis=0)
e28 = LA.norm((dkpre8[:,:99]-truestate8[:, beta0+1:]), axis=0)
e1 = LA.norm((tvdmdstore[:,:199]-truestate[:, beta0+1:]), axis=0)
e2 = LA.norm((dkpre[:,:199]-truestate[:, beta0+1:]), axis=0)

e1bar = np.zeros([e1.shape[0]])
e2bar = np.zeros([e1.shape[0]])
emax = np.zeros(e1.shape[0])
truemax1 = max(truestate[0, beta0+1:])
truemax2 = max(truestate[1, beta0+1:])
truemin1 = min(truestate[1, beta0+1:])
truemin2 = min(truestate[1, beta0+1:])
for i in range(1,e1.shape[0]+1):
    e1bar[i-1] = sum(e1[0:i])/(i)
    e2bar[i-1] = sum(e2[0:i])/(i)
    emax[i-1] = np.sqrt(max((truemax1 - truestate[0, beta0+i]), (truestate[0, beta0+i]-truemin1))**2 
    + max((truemax2 - truestate[1, beta0+i]), (truestate[1, beta0+i]-truemin2))**2)

# plot
plt.rcParams['figure.dpi'] = 100
plt.figure(figsize=(6, 4.3))
plt.subplot(2,1,1)
plt.plot(e18, 'k--.', label='$\\tilde{e}_k, \\gamma=0.8$', linewidth=1.0)
plt.plot(e28, 'b--.', label='$e_k, \\gamma=0.8$', linewidth=1.0)
plt.tick_params(labelsize=12)
plt.xlabel('Iteration Steps', fontsize=12)
plt.ylim([0,max(max(e18),max(e28))+0.25])
plt.legend(loc='upper right', fontsize=10, shadow=True)
plt.subplot(2,1,2)
plt.plot(e1, 'k--.', label='$\\tilde{e}_k, \\gamma=6$', linewidth=1.0)
plt.plot(e2, 'b--.', label='$e_k, \\gamma=6$', linewidth=1.0)
plt.tick_params(labelsize=12)
plt.xlabel('Iteration Steps', fontsize=12)
plt.ylim([0,max(max(e1),max(e2))+0.25])
plt.legend(loc='upper right', fontsize=10, shadow=True)

plt.rcParams['figure.dpi'] = 100
plt.figure(figsize=(6,4.3))
plt.subplot(2,1,1)
plt.plot(truestate[0, beta0+1:], 'r-',
         label='$x_1(k)$', linewidth=1.0)
plt.plot(tvdmdstore[0, :199], 'k--',
         label='$\\tilde{x}_1(k)$', linewidth=2.0)
plt.plot(dkpre[0, :199], 'b--',
         label='$\hat{x}_1(k)$', linewidth=2.0)
plt.tick_params(labelsize=12)
plt.xlabel('Iteration Steps', fontsize=12)
plt.xlim([25,125])
plt.legend(loc='best', fontsize=10, shadow=True)

plt.subplot(2,1,2)
plt.plot(truestate[1, beta0+1:], 'r-',
         label='$x_2(k)$', linewidth=1.0)
plt.plot(tvdmdstore[1, :199], 'k--',
         label='$\\tilde{x}_2(k)$', linewidth=2.0)
plt.plot(dkpre[1, :199], 'b--',
         label='$\hat{x}_2(k)$', linewidth=2.0)
plt.tick_params(labelsize=12)
plt.xlabel('Iteration Steps', fontsize=12)
plt.xlim([25,125])
plt.legend(loc='best', fontsize=10, shadow=True)

plt.show()