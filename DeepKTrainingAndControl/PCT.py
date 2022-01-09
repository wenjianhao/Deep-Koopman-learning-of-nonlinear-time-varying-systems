"""
===================================================================================
Description: Private Controllers Toolbox (PCT)
Author: Wenjian Hao, Bowen Huang
Low-level: Casadi

Version: 2021

How to use?
input state:  x = x - x_goal
MPC:
    1. PCTMPC(A, B, Q, R, Qfinal, controlMax, controlMin, Xmax, Xmin)
    2. compute_policy_gains(Time horizon, x)
Finite LQR:
    1. PCTFILQR(A, B, C, Q, R, N), N: time steps
    2. compute_policy_gains(Time horizon, x)
Infinite LQR:
    1. PCTINFLQR(A, B, C, Q, R)
    2. compute_policy_gains(x)
To Add
...
===================================================================================
"""
import scipy
import control
import scipy.linalg
import numpy as np
import casadi as cs
from casadi import *
from sys import path
path.append(r"C:/Users/14197/anaconda3/envs/mlearning/casadi-windows-py37-v3.5.5-64bit")

#================================
# Model Predictive Control
#================================
class PCTMPC(object):
    def __init__(self, A, B, Q, R, Qf, umax=None, umin=None, xmin=None, xmax=None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.umax = umax
        self.umin = umin
        self.xmin = xmin
        self.xmax = xmax
        ctrb = control.ctrb(self.A, self.B)
        print('System Controllability Rank is: ', np.linalg.matrix_rank(ctrb))

    def compute_policy_gains(self, N, x_t0):
        # Need to stabilize the system around error = 0, command = 0
        # N: Time horizon, x_to: Initial state
        """
        solve MPC with modeling tool for test
        """
        (nx, nu) = self.B.shape
        opti = Opti()
        x = opti.variable(nx, N+1)
        u = opti.variable(nu, N)
        costlist = 0.0

        for t in range(N):
            costlist += 0.5 * cs.mtimes(cs.mtimes(x[:, t].T, self.Q), x[:, t])
            costlist += 0.5 * cs.mtimes(cs.mtimes(u[:, t].T, self.R), u[:, t])            
            opti.subject_to((x[:, t + 1]) == cs.mtimes(self.A, (x[:, t])) + cs.mtimes(self.B, (u[:, t])))
            if self.xmin is not None:
                opti.subject_to((x[:, t]) >= self.xmin[:, 0])
            if self.xmax is not None:
                opti.subject_to((x[:, t]) <= self.xmax[:, 0])

            if self.umin is not None:
                opti.subject_to((u[:, t]) >= self.umin[0])
            if self.umax is not None:
                opti.subject_to((u[:, t]) <= self.umax[0])

        costlist += 0.5 * cs.mtimes(cs.mtimes(x[:, N].T, self.Qf), x[:, N]) # terminal cost
        if self.xmin is not None:
            opti.subject_to((x[:, N]) >= self.xmin[:, 0])
        if self.xmax is not None:
            opti.subject_to((x[:, N]) <= self.xmax[:, 0])

        opti.subject_to((x[:, 0]) == x_t0)
        opti.minimize(costlist)
        # opti.subject_to(constrlist==0)
        opti.solver('ipopt')
        sol = opti.solve()
        # print(sol.value(u))
        # prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)
        # prob.solve(verbose=True)
        # prob.solve(verbose=False)
        return sol.value(x), sol.value(u)

#================================
# LQR with infinite time horizon
#================================
class PCTINFLQR(object):
    def __init__(self, A, B, C, Q, R):
        # set up LQR
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        ctrb = control.ctrb(self.A, self.B)
        print('System Controllability Rank is: ', np.linalg.matrix_rank(ctrb))

        # Lifting the Q matrix and solve K
        Q = self.C.T * self.Q * self.C 
        P = scipy.linalg.solve_discrete_are(self.A, self.B, Q, self.R)
        P = np.matrix(P)
        self.K = np.matrix(scipy.linalg.inv(self.B.T*P*self.B+self.R)*(self.B.T*P*self.A))

    def compute_policy_gains(self, x):
        # get LQR contr
        return -self.K * x

#================================
# LQR with finite time horizon
#================================
class PCTFILQR(object):
    def __init__(self, A, B, C, Q, R, N):
        # N: LQR time steps
        # set up LQR
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.N = N 
        ctrb = control.ctrb(self.A, self.B)
        print('System Controllability Rank is: ', np.linalg.matrix_rank(ctrb))

        # Lifting the Q matrix and solve K
        Q = self.C.T * self.Q * self.C
        self.K_stack = []
        P_ini = Q
        for i in range(self.N):
            self.K = -np.linalg.inv(self.R + self.B.T * P_ini * self.B) * self.B.T * P_ini * self.A
            P_ini = Q + self.K.T * self.R * self.K + (self.A + self.B * self.K).T * P_ini * (self.A + self.B * self.K)
            self.K_stack.append(self.K)

    def compute_policy_gains(self, CurrentTimeStep, x):
        return self.K_stack(self.N-1-CurrentTimeStep)*x
