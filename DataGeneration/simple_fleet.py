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

#==================
# Parameters
#==================
# simulation time steps and trails
sims = 120
trails = 10
# initial states for four models
inim10 = 0
inim11 = 0
u10 = 0
inim30 = 0
inim31 = 0
u30 = 0
inim2 = 0
inim4 = 0
# states array
m1state = np.zeros(sims)
m2state = np.zeros(sims)
m3state = np.zeros(sims)
m4state = np.zeros(sims)
u1 = np.zeros(sims-1)
u2= np.zeros(sims-1)
u3 = np.zeros(sims-1)
u4 = np.zeros(sims-1)
state = np.zeros((8,trails*sims-trails))
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
m1s = []
m2s = []
m3s = []
m4s = []
u1s = []
u2s = []
u3s = []
u4s = []
for j in range(trails):
    for i in range(sims):
        if i >= 2:
            u1t = np.random.uniform(low=-1, high=3)
            u3t = np.random.uniform(low=-1, high=3)
            m1state[i] = m1state[i-2] * m1state[i-1] / (1 + m1state[i-2]**2 + m1state[i-1]**2) + 3*u1t
            m3state[i] = (m3state[i-2] * m3state[i-1] * u3t + u3t)/(1+m3state[i-2]**2 + m3state[i-1]**2)
            u1[i-1] = u1t
            u3[i-1] = u3t
        if i >= 1:
            u2t = np.random.uniform(low=-1, high=2)
            u4t = np.random.uniform(low=-1, high=3)
            m2state[i] = m2state[i-1]/(1+m2state[i-1]**4) + u2t**4
            m4state[i] = m4state[i-1] * u4t / (1+m4state[i-1]**4) +2*u4t
            u2[i-1] = u2t
            u4[i-1] = u4t
    m1s.append(m1state[0:(sims-1)])
    m2s.append(m2state[0:(sims-1)])
    m3s.append(m3state[0:(sims-1)])
    m4s.append(m4state[0:(sims-1)])
    u1s.append(u1)
    u2s.append(u2)
    u3s.append(u3)
    u4s.append(u4)

# plot one game traj
fig0 = plt.figure(figsize=(8,6))
plt.subplot(1,1,1)
# plt.plot(m1s[3], label='agent1')
plt.plot(m1state, label='agent1')
plt.plot(m2state, label='agent2')
plt.plot(m3state, label='agent3')
plt.plot(m4state, label='agent4')
plt.xlabel("time steps",fontsize=12)
plt.ylabel("position",fontsize=12)
plt.legend(loc='upper right', prop={'size': 8})
plt.title('One trail trajectory during data collecting',fontsize=12,color='black')

state[0,:] = np.squeeze(np.array([m1s]).reshape(1,-1))
state[1,:] = np.squeeze(np.array([m2s]).reshape(1,-1))
state[2,:] = np.squeeze(np.array([m3s]).reshape(1,-1))
state[3,:] = np.squeeze(np.array([m4s]).reshape(1,-1))
state[4,:] = np.squeeze(np.array([u1s]).reshape(1,-1))
state[5,:] = np.squeeze(np.array([u2s]).reshape(1,-1))
state[6,:] = np.squeeze(np.array([u3s]).reshape(1,-1))
state[7,:] = np.squeeze(np.array([u4s]).reshape(1,-1))

filename = 'data/trainingdata.pkl'
joblib.dump(state, filename)
print('Saved data, data dimension: 8 X', trails*sims)
plt.show()
