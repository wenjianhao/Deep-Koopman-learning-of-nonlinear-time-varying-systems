"""
Model predictive control sample code with modeling tool (cvxpy)


"""
import scipy
import cvxpy
import numpy as np
import scipy.linalg


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
        x = cvxpy.Variable((nx, N + 1))
        u = cvxpy.Variable((nu, N))

        costlist = 0.0
        constrlist = []

        for t in range(N):
            costlist += 0.5 * cvxpy.quad_form((x[:, t]), self.Q)
            costlist += 0.5 * cvxpy.quad_form((u[:, t]), self.R)

            constrlist += [x[:, t + 1] == self.A * (x[:, t]) + self.B * ((u[:, t])-u_0)]

            if self.xmin is not None:
                constrlist += [(x[:, t]) >= self.xmin[:, 0]]
            if self.xmax is not None:
                constrlist += [(x[:, t]) <= self.xmax[:, 0]]

            if self.umin is not None:
                # constrlist += [(u[:, t]+u_0) >= self.umin[:, 0]]  # input constraints
                constrlist += [(u[:, t]) >= self.umin[0]]  # input constraints
            if self.umax is not None:
                # constrlist += [(u[:, t]+u_0) <= self.umax[:, 0]]  # input constraints, multiple inputs
                constrlist += [(u[:, t]) <= self.umax[0]]  # input constraints, multiple inputs


        costlist += 0.5 * cvxpy.quad_form((x[:, N]), self.Qf)  # terminal cost
        if self.xmin is not None:
            constrlist += [(x[:, N]) >= self.xmin[:, 0]]
        if self.xmax is not None:
            constrlist += [(x[:, N]) <= self.xmax[:, 0]]

        # if self.umax is not None:
            # constrlist += [u <= self.umax]  # input constraints
        # if self.umin is not None:
            # constrlist += [u >= self.umin]  # input constraints

        constrlist += [(x[:, 0]) == x_t0]  # inital state constraints

        prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)

        # prob.solve(verbose=True)
        prob.solve(verbose=False)

        return x.value, u.value
        