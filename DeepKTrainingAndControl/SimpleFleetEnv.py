'''
Project: Data driven control for the multiagent fleet
Description: Simple multiagent models simulation for time-series data collection
Author: Wenjian Hao
Date: Sep/2021
Models in this file comes from paper: https://ieeexplore.ieee.org/document/9261580
'''

import joblib
import numpy as np 
import matplotlib.pyplot as plt

# MPC controller
class SimpleFleet(object):
    def __init__(self, initial_states, simtime):
        #===============================================
        # set parameters
        #===============================================
        '''
        inital states form: agent1 states t0, t1, agent2 states t0, agent3 states t0, t1, agent4 states t0
        agent1 control t0, agent3 control t0
        '''
        self.instates = initial_states 
        self.simtime = simtime
        self.state1 = np.zeros(simtime)
        self.state2 = np.zeros(simtime)     
        self.state3 = np.zeros(simtime) 
        self.state4 = np.zeros(simtime) 
        self.control1 = np.zeros(simtime) 
        self.control2 = np.zeros(simtime)  
        self.control3 = np.zeros(simtime) 
        self.control4 = np.zeros(simtime) 
        self.state1[0] = self.instates[0]
        self.state1[1] = self.instates[1]
        self.state2[0] = self.instates[2]
        self.state3[0] = self.instates[3]
        self.state3[1] = self.instates[4]
        self.state4[0] = self.instates[5]
        self.control1[0] = self.instates[6]
        self.control3[0] = self.instates[7]

    def sim(self, u, currentsteps):
        u = np.squeeze(u)
        if currentsteps >= 2:
            u1t = u[0]
            u3t = u[2]
            self.state1[currentsteps] = self.state1[currentsteps-2] * self.state1[currentsteps-1] / (1 + (self.state1[currentsteps-2])**2 + (self.state1[currentsteps-1])**2) + 3*u1t
            self.state3[currentsteps] = (self.state3[currentsteps-2] * self.state3[currentsteps-1] * u3t + u3t)/(1+(self.state3[currentsteps-2])**2 + (self.state3[currentsteps-1])**2)
        if currentsteps >= 1:
            u2t = u[1]
            u4t = u[3]
            self.state2[currentsteps] = self.state2[currentsteps-1]/(1+(self.state2[currentsteps-1])**4) + (u2t)**4
            self.state4[currentsteps] = self.state4[currentsteps-1] * u4t / (1+(self.state4[currentsteps-1])**4) +2*u4t
        if currentsteps > 0 and (self.state1[currentsteps] < 0 or self.state2[currentsteps] < 0 or self.state3[currentsteps] < 0 or self.state4[currentsteps] < 0 \
            or currentsteps > self.simtime \
                or self.state1[currentsteps] == self.state2[currentsteps] !=0 or self.state1[currentsteps] == self.state3[currentsteps]!=0 \
                    or self.state1[currentsteps] == self.state4[currentsteps]!=0 or self.state2[currentsteps] == self.state3[currentsteps]!=0 \
                        or self.state2[currentsteps] == self.state4[currentsteps]!=0 or self.state3[currentsteps] == self.state4[currentsteps]!=0):
            done = True
            print('state', [self.state1[currentsteps], self.state2[currentsteps], self.state3[currentsteps], self.state4[currentsteps]])
        else:
            done = False

        return self.state1[currentsteps], self.state2[currentsteps], self.state3[currentsteps], self.state4[currentsteps], done

    def reset(self):
        self.state1 = []
        self.state2 = []       
        self.state3 = [] 
        self.state4 = [] 
        self.control1 = [] 
        self.control2 = [] 
        self.control3 = [] 
        self.control4 = [] 
        self.state1.append(self.instates[0])
        self.state1.append(self.instates[1])
        self.state2.append(self.instates[2])
        self.state3.append(self.instates[3])
        self.state3.append(self.instates[4])
        self.state4.append(self.instates[5])
        self.control1.append(self.instates[6])
        self.control3.append(self.instates[7])