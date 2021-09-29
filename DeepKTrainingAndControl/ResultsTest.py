'''
===================================================================================
Description: Result Test
Project: Data driven control for the multiagent fleet
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
checkpoint = torch.load(load_name)
model = LNN()
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
        time_horizon = 6
        time_execute = 1
        self.A = np.matrix(A)
        self.B = np.matrix(B)
        self.C = np.matrix(C)        
        # define weighting structure
        Q = np.diag([2,2,2,2])
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
        Ngames = 6
        traj = []
        liftstate = []
        
        for i in range(Ngames):
            done = False
            score = 0
            step = 0
            traj_each = []
            lifteach = []
            while not done:
                for k in range(time_execute):
                    # get current state
                    cur_state = np.array([
                        observation[0],
                        observation[1],
                        observation[2]
                        ])
                    # lift current states
                    cur_input_tensor = torch.from_numpy(cur_state.T).float()
                    cur_state_lifted = model.forward(cur_input_tensor).cpu().detach().numpy()
                    cur_state_lifted = cur_state_lifted
                    # mpc planning
                    if k==0:
                        try:
                            x_mpc, u_mpc = mpc.compute_policy_gains(N=time_horizon,x_t0=cur_state_lifted.T)
                        except:
                            done = True
                    # opt_u_list = u_mpc[0][k]
                    opt_u_list = u_mpc[0]
                    # record data
                    lifteach.append(cur_state_lifted)
                    # step the game
                    s1, s2, s3, s4, done = self.env.step([opt_u_list], step)
                    print(opt_u_list)     
                    self.total_steps += 1                
                    step += 1

            numgames.append(i)
            gamereward.append(score)
            stepreward.append(reward)
            totalsteps.append(self.total_steps)
            traj.append(traj_each)
            liftstate.append(lifteach)

        # plot reward according to behavior
        print('reward mean is: ', np.array(gamereward).mean())
        #================================================================================
        # Figures Plotting
        #================================================================================
        trajectories = np.array(traj)
        # trajectories = trajectories.T
        traj_plot = False
        if traj_plot:
            gamereward = np.array(gamereward).reshape(-1,1)
            fig0 = plt.figure(figsize=(8,6))
            plt.subplot(1,1,1)
            plt.plot(numgames, gamereward, label='trend')
            plt.xlabel("game",fontsize=12)
            plt.ylabel("reward",fontsize=12)
            # plt.title('Reward of each game by deploying Deep_EDMD',fontsize=12,color='r')
            # 3d cos sin and theta dot
            fig200 = plt.figure(figsize=(6,9))
            for j in range(Ngames):
                # trajectory = np.array(trajectories[j])
                theta20 = []
                dtheta20 = []
                ctr = []
                time = []
                time_step = 0
                for s in trajectories[j]:
                    if s[1] < 0 and s[0] < 0:
                        realtheta = -np.arccos(s[0]) + 2*np.pi # map with '+2pi'
                    if s[1] < 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                        if realtheta < -0.4:
                            realtheta = np.arcsin(s[1]) + 2*np.pi # map with '+2pi'
                    if s[1] > 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                    if s[1] > 0 and s[0] < 0:
                        realtheta = np.arccos(s[0])           
                    if s[1] == 0 and s[0] == 1:
                        realtheta = 0
                    if s[1] == 0 and s[0] == -1:
                        realtheta = np.pi
                    if s[1] == 1 and s[0] == 0:
                        realtheta = np.pi/2
                    if s[1] == -1 and s[0] == 0:
                        realtheta = -np.pi/2 + 2*np.pi # map with '+2pi'
                    
                    theta20.append(realtheta)
                    dtheta20.append(s[2])
                    time.append(time_step)
                    time_step += 1
                    ctr.append(s[3])

                plt.subplot(3,1,1)
                plt.plot(time, theta20)#, c='r')
                plt.xlabel('time', fontsize=12)#, color='r')
                plt.ylabel('theta', fontsize=12)

                plt.subplot(3,1,2)
                plt.plot(time, dtheta20)#, c='r')
                plt.xlabel('time', fontsize=12)#, color='r')
                plt.ylabel('theta dot', fontsize=12)

                plt.subplot(3,1,3)
                plt.plot(time, ctr)#, c='r')
                plt.xlabel('time', fontsize=12)#, color='r')
                plt.ylabel('control', fontsize=12)

            # # 3d cos sin and theta dot
            # fig201= plt.figure(figsize=(6,6))
            # for j in range(Ngames):
            #     # trajectory = np.array(trajectories[j])
            #     ctr = []
            #     time = []
            #     time_step = 0
            #     for s in trajectories[j]:
            #         ctr.append(s[3])
            #         time.append(time_step)
            #         time_step += 1

            #     plt.subplot(1,1,1)
            #     plt.plot(time, ctr)#, c='r')
            #     plt.xlabel('time', fontsize=12)#, color='r')
            #     plt.ylabel('control', fontsize=12)


            
            fig00 = plt.figure(figsize=(6,6))
            for i in range(Ngames):
                theta = []
                velo = []
                energy = []
                for s in trajectories[i]:
                    if s[1] < 0 and s[0] < 0:
                        realtheta = -np.arccos(s[0]) #+ 2*np.pi # map with '+2pi'
                    if s[1] < 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                        # if realtheta < -0.4:
                        #     realtheta = np.arcsin(s[1]) + 2*np.pi # map with '+2pi'
                    if s[1] > 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                    if s[1] > 0 and s[0] < 0:
                        realtheta = np.arccos(s[0])           
                    if s[1] == 0 and s[0] == 1:
                        realtheta = 0
                    if s[1] == 0 and s[0] == -1:
                        realtheta = np.pi
                    if s[1] == 1 and s[0] == 0:
                        realtheta = np.pi/2
                    if s[1] == -1 and s[0] == 0:
                        realtheta = -np.pi/2 #+ 2*np.pi # map with '+2pi'
                    ge = 0.5*s[2]**2 + s[0] + s[3]
                    theta.append(realtheta)
                    velo.append(s[2])
                    energy.append(ge)
                plt.subplot(1,1,1)
                plt.scatter(theta,velo,c=energy, cmap='jet')
                im00 = plt.scatter(theta, velo, c=energy, cmap='jet')
                plt.xlim((-np.pi,np.pi))
                # plt.xlim((-1,2*np.pi))
                plt.ylim((-8,8))
                plt.xlabel('theta',fontsize=10)
                plt.ylabel('theta dot',fontsize=10)
                # plt.title('2D Trajectories',fontsize=25)
                plt.plot(theta[0],velo[0], marker='*', markersize=16,c='r', label='start point') 
                plt.plot(0,0, marker='*', markersize=16,c='black',label='goal position')
            fig00.colorbar(im00)

            # 3d cos sin and theta dot
            fig2 = plt.figure(figsize=(8,6))
            ax = fig2.add_subplot(111, projection='3d')
            for j in range(Ngames):
                # trajectory = np.array(trajectories[j])
                cos = []
                sin = []
                tvelo = []
                for point in trajectories[j]:
                    cos.append(point[0])
                    sin.append(point[1])
                    tvelo.append(point[2])
                ax.scatter(cos,sin,tvelo, s=100)#, c='r')
                ax.plot(cos,sin,tvelo,linewidth=8)#, color='r')
                ax.scatter(1,0,0, color='r',s=600)
                ax.set_xlim3d(-1,1)
                ax.set_ylim3d(-1,1)
                ax.set_zlim3d(-8,8)
                ax.set_zlabel('theta_dot',fontsize=20)
                ax.set_ylabel('sin(theta)',fontsize=20)
                ax.set_xlabel('cos(theta)',fontsize=20)
                ax.set_title('3D trajectories',fontsize=25)
            # 3d theta theta dot and control
            fig3 = plt.figure(figsize=(12,6))
            ax = fig3.add_subplot(121, projection='3d')
            for k in range(Ngames):
                # trajectory = np.array(trajectories[j])
                theta_plt = []
                dtheta = []
                self.ctr = []
                self.energy = []
                for s in trajectories[k]:
                    if s[1] < 0 and s[0] < 0:
                        realtheta = -np.arccos(s[0]) + 2*np.pi # map with '+2pi'
                    if s[1] < 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                        if realtheta < -0.4:
                            realtheta = np.arcsin(s[1]) + 2*np.pi # map with '+2pi'
                    if s[1] > 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                    if s[1] > 0 and s[0] < 0:
                        realtheta = np.arccos(s[0])           
                    if s[1] == 0 and s[0] == 1:
                        realtheta = 0
                    if s[1] == 0 and s[0] == -1:
                        realtheta = np.pi
                    if s[1] == 1 and s[0] == 0:
                        realtheta = np.pi/2
                    if s[1] == -1 and s[0] == 0:
                        realtheta = -np.pi/2 + 2*np.pi # map with '+2pi'
                    ge = 0.5*s[2]**2 + s[0] + s[3]
                    theta_plt.append(realtheta)
                    dtheta.append(s[2])
                    self.ctr.append(s[3])
                    self.energy.append(ge)
                ax.plot_trisurf(theta_plt,dtheta,self.energy,cmap=plt.cm.winter)#,cmap=plt.cm.Spectral)#, c='r')
                # ax.scatter(theta_plt,dtheta,self.energy)
            ax.plot(theta_plt,dtheta,self.energy,linewidth=12)#, color='r')
            ax.scatter(0,0,ge, color='r',s=600)
            ax.set_zlabel('energy',fontsize=20)
            ax.set_ylabel('theta dot',fontsize=20)
            ax.set_xlabel('theta',fontsize=20)
            ax.set_title('3d plot of theta, theta dot and energy',fontsize=25)

            # 3d cos sin and theta dot
            fig20 = plt.figure(figsize=(9,6))
            ax = fig20.add_subplot(111, projection='3d')
            for j in range(Ngames):
                # trajectory = np.array(trajectories[j])
                theta20 = []
                dtheta20 = []
                time = []
                time_step = 0
                for s in trajectories[j]:
                    if s[1] < 0 and s[0] < 0:
                        realtheta = -np.arccos(s[0]) + 2*np.pi # map with '+2pi'
                    if s[1] < 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                        if realtheta < -0.4:
                            realtheta = np.arcsin(s[1]) + 2*np.pi # map with '+2pi'
                    if s[1] > 0 and s[0] > 0:
                        realtheta = np.arcsin(s[1])
                    if s[1] > 0 and s[0] < 0:
                        realtheta = np.arccos(s[0])           
                    if s[1] == 0 and s[0] == 1:
                        realtheta = 0
                    if s[1] == 0 and s[0] == -1:
                        realtheta = np.pi
                    if s[1] == 1 and s[0] == 0:
                        realtheta = np.pi/2
                    if s[1] == -1 and s[0] == 0:
                        realtheta = -np.pi/2 + 2*np.pi # map with '+2pi'
                    
                    theta20.append(realtheta)
                    dtheta20.append(s[2])
                    time.append(time_step)
                    time_step += 1
                ax.scatter(time, theta20, dtheta20, s=100)#, c='r')
                im201 = ax.scatter(time, theta20, dtheta20, s=100, c=self.energy, cmap='jet')#, c='r')
                ax.plot(time, theta20, dtheta20, linewidth=6)#, color='r')
                ax.scatter(time_step,0,0,color='r',s=600)
                ax.set_zlabel('theta_dot',fontsize=20)
                ax.set_ylabel('theta',fontsize=20)
                ax.set_xlabel('time',fontsize=20)
            fig20.colorbar(im201)
            # plot psi1 psi2 
            ax = fig3.add_subplot(122, projection='3d')
            for j in range(Ngames):
                psi6 = []
                psi7 = []
                for s in liftstate[j]:
                    psi6.append(s[5])
                    psi7.append(s[6])
                
                ax.plot_trisurf(psi6, psi7, self.energy, cmap=plt.cm.winter)
                # ax.scatter(psi6,psi7,self.energy, s=100)
            ax.plot(psi6,psi7,self.energy,linewidth=12)
            ax.scatter(s[5],s[6], color='r',s=600)
            ax.set_zlabel('control',fontsize=20)
            ax.set_ylabel('psi2',fontsize=20)
            ax.set_xlabel('psi1',fontsize=20)
            ax.set_title('3d plot of psi1, psi2 and control',fontsize=25)

        plt.show()
        self.env.close()

# choose a controller
if __name__=='__main__':
    # run = FINITE_LQR()
    # run = INFINITE_LQR()
    run = MPC()
