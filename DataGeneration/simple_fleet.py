'''
Project: Data driven control for the multiagent fleet
Description: Simple multiagent models simulation for time-series data collection
Author: Wenjian Hao
Date: Sep/2021

Models in this file comes from paper: https://ieeexplore.ieee.org/document/9261580

'''
import numpy as np 
import joblib


#==================
# Parameters
#==================
# simulation time
sims = 1000 
# initial states for four models
inim10 = 10
inim11 = 9
inim30 = 10
inim31 = 9
inim2 = 10
inim4 = 10
# states array
m1state = np.zeros((1,sims))
m2state = np.zeros((1,sims))
m3state = np.zeros((1,sims))
m4state = np.zeros((1,sims))
state = np.zeros((1,sims))
m1state[0] = inim10
m1state[1] = inim11
m2state[0] = inim2
m3state[0] = inim30
m3state[1] = inim31
m4state[0] = inim4

#==================
# simulation
#==================
for i in range(sims):
    if i >= 2:
        m1state[i] = m1state[i-2] * m1state[i-1] / (1 + m1state[i-2]**2 + m1state[i-1]**2) + 3*np.random(low=-5, high=5)
