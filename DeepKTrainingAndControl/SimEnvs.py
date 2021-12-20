'''
Project: Project: Deep Koopman Representation for Time-varying Systems
Description: Simulation Environments
Author: Wenjian Hao
Date: Dec/2021
'''

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ToyEx(object):
    def __init__(self, inistate, simtime, dt):
        '''
        This sim env is a very simple time varying linear dynamical system
        x(t)_dot = A(t) x(t), where, A(t) = [0, 1+w*t; -(1+w*t), 0]
        '''
        # parameters
        self.epsilon = 1e-1
        self.tspan = np.linspace(0, simtime, int(simtime/dt+1))
        self.dt = dt
        self.x0 = inistate

    def dyn(self, x, t):
        x1, x2 = x
        # x2 = np.cos(x2)
        # t = np.cos(t)
        dxdt = [(1+self.epsilon*t)*x2, -(1+self.epsilon*t)*x1]
        # dxdt = [(1+epsilon*np.cos(t))*x2, -(1+epsilon*t)*x1]
        return dxdt

    def sim(self):
        xsol = odeint(self.dyn, self.x0, self.tspan).T
        # extract snapshots
        x, y = xsol[:, :-1], xsol[:, 1:]
        t = self.tspan[1:]
        # true dynamics, true eigenvalues
        n, m = len(x[:, 0]), len(x[0, :])
        A = np.empty((n, n, m))
        evals = np.empty((n, m), dtype=complex)
        for k in range(m):
            A[:, :, k] = np.array([[0, (1+self.epsilon*t[k])], [-(1+self.epsilon*t[k]), 0]])
            evals[:, k] = np.linalg.eigvals(A[:, :, k])
        return x, y, evals
