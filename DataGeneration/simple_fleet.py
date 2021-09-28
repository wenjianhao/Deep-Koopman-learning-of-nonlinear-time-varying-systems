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
# simulation time steps and trails
sims = 500
trails = 10
# initial states for four models
inim10 = 10
inim11 = 9
u10 = 0
inim30 = 10
inim31 = 9
u30 = 0
inim2 = 10
inim4 = 10
# states array
m1state = np.zeros(trails*sims)
m2state = np.zeros(trails*sims)
m3state = np.zeros(trails*sims)
m4state = np.zeros(trails*sims)
u1 = np.zeros(trails*sims-1)
u2= np.zeros(trails*sims-1)
u3 = np.zeros(trails*sims-1)
u4 = np.zeros(trails*sims-1)
state = np.zeros((8,trails*sims))
m1state[0] = inim10
m1state[1] = inim11
m2state[0] = inim2
m3state[0] = inim30
m3state[1] = inim31
m4state[0] = inim4
u1[0] = u10
u3[0] = u30

#==================
# simulation
#==================
for i in range(sims):
    if i >= 2:
        u1 = np.random.uniform(low=-5, high=5)
        u3 = np.random.uniform(low=-5, high=5)
        m1state[i] = m1state[i-2] * m1state[i-1] / (1 + m1state[i-2]**2 + m1state[i-1]**2) + 3*u1
        m3state[i] = (m3state[i-2] * m3state[i-1] * u3 + u3)/(1+m3state[i-2]**2 + m3state[i-1]**2)
        u1[i-1] = u1
        u3[i-1] = u3
    if i >= 1:
        u2 = np.random.uniform(low=-5, high=5)
        u4 = np.random.uniform(low=-5, high=5)
        m2state[i] = m2state[i-1]/(1+m2state[i-1]**4) + u2**4
        m4state[i] = m4state[i-1] * u4 / (1+m4state[i-1]**4) +2*u4
        u2[i-1] = u2
        u4[i-1] = u4
