'''
Description: Paper codes of 'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." 
                            Automatica 159 (2024): 111372.'

Author: Wenjian Hao, Purdue University

Start at: Sep 2021
Last Revision: Jan 2022

Training data format:
 | x_1(1) x_1(2) ... x_1(t) |
 | x_2(1) x_1(2) ... x_1(t) |
 | ...    ...    ...    ... |
 | x_n(1) x_n(2) ... x_n(t) |
 | u_1(1) u_1(2) ... u_1(t) |
 | ...    ...    ...    ... |
 | u_m(1) u_m(2) ... u_m(t) |
 where,
 n: the dimension of the system states
 m: the dimension of the control inputs
 t: time index, t = 0, 1, 2, ...
'''

#=====================
# Load third packages
#=====================
import config
import joblib

import numpy as np

from SimEnvs import ToyEx
from odmd import OnlineDMD
from utils import DKTV_training

if __name__ == "__main__":
    #=====================
    # data generation
    #=====================
    x0                = [1, 0]
    dt                = 0.1
    SimTime           = 30
    env               = ToyEx(x0, SimTime, dt)
    x, y, groundtruth = env.sim()
    # split data
    train_len     = int(len(x[0]))
    totalen       = len(x[0])
    train_samples = x[:, 0:(train_len)]
    train_label   = y[:, 0:(train_len)]
    # recording
    Astack      = np.empty((config.dimensions['dim_lifting'], config.dimensions['dim_lifting'], len(train_samples[0])))
    evaleigenv  = np.empty((config.dimensions['dim_states'], len(train_samples[0])), dtype=complex)
    tspan       = np.linspace(0, SimTime, int(SimTime/dt+1))
    plott       = tspan[1:]
    numdyn      = int(np.floor((train_len - config.training_parameters['prebatch_size'])/config.training_parameters['batch_size'])+1)
    evaleigenv1 = np.empty((config.dimensions['dim_states'], numdyn), dtype=complex)
    Astk        = np.empty((config.dimensions['dim_lifting'], config.dimensions['dim_lifting'], numdyn))
    Cstk        = np.empty((config.dimensions['dim_states'], config.dimensions['dim_lifting'], numdyn))
    Hisstk      = np.empty((config.dimensions['dim_lifting'], config.dimensions['dim_lifting'], numdyn))
    tvdmdstore  = []

    #================================
    # comparison algorithm of TVDMD
    #================================
    odmd = OnlineDMD(config.dimensions['dim_states'], 1.0)
    odmd.initialize(x[:, :config.training_parameters['prebatch_size']], y[:, :config.training_parameters['prebatch_size']])
    for k in range(config.training_parameters['prebatch_size'], len(train_samples[0])):
        odmd.update(x[:, k], y[:, k])
        TVDMDpred = np.matrix(odmd.A) * np.matrix(x[:,k]).T
        tvdmdstore.append(TVDMDpred)
    tvdmdstore = np.array(tvdmdstore).squeeze().T
    joblib.dump(x, config.files_dir['xfile'])
    joblib.dump(tvdmdstore, config.files_dir['tvdpred'])

    #=================================
    # DKTV (the proposed algorithm)
    #=================================
    # initialization
    DKTV          = DKTV_training(config)
    A0, C0, His0  = DKTV.pretrain_model(train_samples[:, 0:config.training_parameters['prebatch_size']+1], train_label[:, 0:config.training_parameters['prebatch_size']+1])
    Astk[:,:,0]   = A0
    Cstk[:,:,0]   = C0
    Hisstk[:,:,0] = His0
    # main algorithm
    dktvtemp = np.zeros((config.dimensions['dim_states'],1)) # record the prediction data
    DKTV     = DKTV_training(config)
    for nd in range(1, numdyn):
        tsam           = train_samples[:, 0:(config.training_parameters['prebatch_size']+nd*config.training_parameters['batch_size'])]
        tlab           = train_label[:, 0:(config.training_parameters['prebatch_size']+nd*config.training_parameters['batch_size'])]
        Ak, Ck, Hisk   = DKTV.DKTV(Astk, Cstk, Hisstk, tsam, tlab, config.training_parameters['batch_size'], nd)
        Astk[:,:,nd]   = Ak
        Cstk[:,:,nd]   = Ck
        Hisstk[:,:,nd] = Hisk
    # Save dynamics stacks for plotting
    joblib.dump(Astk, config.files_dir['fileastack'])
    joblib.dump(Cstk, config.files_dir['filecstack'])
    joblib.dump(Hisstk, config.files_dir['filehiscstack'])
    
    print('Finished learning, please run plot_compare.py for results visualization')
