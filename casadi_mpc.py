"""
Model predictive control sample code with modeling tool (cvxpy)


"""
import scipy
# import cvxpy
import numpy as np
import scipy.linalg
from sys import path
path.append(r"C:/Users/14197/anaconda3/envs/mlearning/casadi-windows-py37-v3.5.5-64bit")
from casadi import *
import casadi as cs


class MPC_CVX(object):
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

        
    def compute_policy_gains(self, N, x_t0, x_lift_0, u_0):
        # Need to stabilize the system around error = 0, command = 0
        """
        solve MPC with modeling tool for test
        """
        (nx, nu) = self.B.shape

        # mpc calculation
        # x = Variable((nx, N + 1))
        # u = Variable((nu, N))

        opti = Opti()
        x = opti.variable(nx, N+1)
        u = opti.variable(nu, N)

        costlist = 0.0
        constrlist = []

        for t in range(N):
            costlist += 0.5 * cs.mtimes(cs.mtimes(x[:, t].T, self.Q), x[:, t])
            costlist += 0.5 * cs.mtimes(cs.mtimes(u[:, t].T, self.R), u[:, t])

            # constrlist += [x[:, t + 1] == self.A * (x[:, t]) + self.B * ((u[:, t])-u_0)]
            
            opti.subject_to((x[:, t + 1]) == cs.mtimes(self.A, (x[:, t])) + cs.mtimes(self.B, ((u[:, t])-u_0)))
            if self.xmin is not None:
                # constrlist += [(x[:, t]) >= self.xmin[:, 0]]
                opti.subject_to((x[:, t]) >= self.xmin[:, 0])
            if self.xmax is not None:
                # constrlist += [(x[:, t]) <= self.xmax[:, 0]]
                opti.subject_to((x[:, t]) <= self.xmax[:, 0])

            if self.umin is not None:
                # constrlist += [(u[:, t]) >= self.umin[0]]  # input constraints
                opti.subject_to((u[:, t]) >= self.umin[0])
            if self.umax is not None:
                # constrlist += [(u[:, t]) <= self.umax[0]]  # input constraints, multiple inputs
                opti.subject_to((u[:, t]) <= self.umax[0])


        costlist += 0.5 * cs.mtimes(cs.mtimes(x[:, N].T, self.Qf), x[:, N]) # terminal cost
        if self.xmin is not None:
            # constrlist += [(x[:, N]) >= self.xmin[:, 0]]
            opti.subject_to((x[:, N]) >= self.xmin[:, 0])
        if self.xmax is not None:
            # constrlist += [(x[:, N]) <= self.xmax[:, 0]]
            opti.subject_to((x[:, N]) <= self.xmax[:, 0])

        # if self.umax is not None:
            # constrlist += [u <= self.umax]  # input constraints
        # if self.umin is not None:
            # constrlist += [u >= self.umin]  # input constraints

        # constrlist += [(x[:, 0]) == x_t0]  # inital state constraints
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

