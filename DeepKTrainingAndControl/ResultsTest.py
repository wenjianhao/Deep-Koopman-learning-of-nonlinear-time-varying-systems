'''
===================================================================================
Description: Result Test
Project: Deep Koopman Representation for Time-varying Systems
Author: Wenjian Hao, Purdue University

Version: Sep / 2021
===================================================================================
'''

import torch
import control

import numpy as np
import matplotlib.pyplot as plt

import joblib
from mpl_toolkits.mplot3d import Axes3D
from LNN import LNN
from PCT import PCTMPC
from SimpleFleetEnv import SimpleFleet

# load files and model
load_name = 'SavedResults/liftnetwork.pth'
# checkpoint = torch.load(load_name)
checkpoint = torch.load(load_name,map_location=torch.device('cpu'))
model = LNN(4,16)
model.load_state_dict(checkpoint['model_lifting'])
A = joblib.load('SavedResults/A.pkl')
B = joblib.load('SavedResults/B.pkl')
C = joblib.load('SavedResults/C.pkl')

# MPC controller
class TEST():
    def __init__(self):
        #===============================================
        # set parameters
        #===============================================
        NUM_STATE = 2
        time_horizon = 20
        time_execute = 1
        self.A = np.matrix(A)
        self.B = np.matrix(B)
        self.C = np.matrix(C)        
        # define weighting structure
        Q = np.diag([2,3,3,3])
        Q_lifted = self.C.T .dot(Q).dot(self.C)       
        Qf = Q_lifted * 2 # weight matrix for terminal state
        R = np.diag([1,1,1,1])
        Ctrb = control.ctrb(self.A,self.B)
        print('Controlability test of A&B:\n Rank of A: {rankA}\n Rank of Ctrb: {rankC}\n Controllable? --> {ctrb}'.format(rankA=self.A.shape[0], rankC=np.linalg.matrix_rank(Ctrb), ctrb=(np.linalg.matrix_rank(Ctrb)==self.A.shape[0])))
        # set control limitation
        umax=np.array([3])
        umin=np.array([-1])
        mpc = PCTMPC(self.A, self.B, Q_lifted, R, Qf, umax=umax, umin=umin)

        # build environment
        sims = 200
        # initial states for four models
        inim10 = 0
        inim11 = 0
        u10 = 0
        inim30 = 0
        inim31 = 0
        u30 = 0
        inim2 = 0
        inim4 = 0
        inistate = np.array([inim10, inim11, inim2, inim30, inim31, inim4, u10, u30])
        self.env = SimpleFleet(inistate, sims)
        self.total_steps = 0
        stepreward = []
        totalsteps = []
        gamereward = []
        numgames = []
        Ngames = 1
        traj = []
        liftstate = []
        
        for i in range(Ngames):
            done = False
            score = 0
            step = 0
            traj_each = []
            lifteach = []
            s1, s2, s3, s4 = np.array([inim10, inim2-inim10, inim30-inim10, inim4-inim10])
            cur_state = np.array([s1, s2-s1, s3-s1, s4-s1])
            goalstate = np.array([5,10,2,2])
            goalstate = torch.from_numpy(goalstate.T).float()
            goal_state_lifted = model.forward(goalstate).cpu().detach().numpy()
            while not done and step < sims:
                for k in range(time_execute):
                    # lift current states
                    cur_input_tensor = torch.from_numpy(cur_state.T).float()
                    cur_state_lifted = model.forward(cur_input_tensor).cpu().detach().numpy()
                    cur_state_lifted = cur_state_lifted - goal_state_lifted
                    # mpc planning
                    if k==0:
                        try:
                            x_mpc, u_mpc = mpc.compute_policy_gains(N=time_horizon,x_t0=cur_state_lifted.T)
                        except:
                            done = True
                    # opt_u_list = u_mpc[0][k]
                    opt_u_list = u_mpc[:,0]
                    # record data
                    lifteach.append(cur_state_lifted)
                    # step the game
                    s1, s2, s3, s4, done = self.env.sim([opt_u_list], step)
                    cur_state = np.array([s1, s2-s1, s3-s1, s4-s1])
                    traj_each.append(cur_state)
                    print(opt_u_list)     

                    self.total_steps += 1                
                    step += 1

            numgames.append(i)
            gamereward.append(score)
            totalsteps.append(self.total_steps)
            traj.append(traj_each)
            liftstate.append(lifteach)
    
        #================================================================================
        # Figures Plotting
        #================================================================================
        trajectories = np.array(traj)
        # trajectories = trajectories.T
        traj_plot = True
        if traj_plot:
            fig1 = plt.figure(figsize=(8,6))
            for j in range(Ngames):
                # trajectory = np.array(trajectories[j])
                agent1 = []
                agent2 = []
                agent3 = []
                agent4 = []
                for s in trajectories[j]:
                    agent1.append(s[0])
                    agent2.append(s[1]+s[0])
                    agent3.append(s[2]+s[0])
                    agent4.append(s[3]+s[0])

                plt.subplot(1,1,1)
                # plt.plot(m1s[3], label='agent1')
                plt.plot(agent1, label='agent1')
                plt.plot(agent2, label='agent2')
                plt.plot(agent3, label='agent3')
                plt.plot(agent4, label='agent4')
                plt.xlabel("time steps",fontsize=12)
                plt.ylabel("agent position",fontsize=12)
                plt.legend(loc='upper right', prop={'size': 8})
                plt.title('One Test Trail Trajectory',fontsize=12,color='black')

        plt.show()

# choose a controller
if __name__=='__main__':
    run = TEST()
