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
from SimpleFleetEnv import SimpleFleet

#==================
# Parameters
#==================
# simulation time steps and trails
sims = 200
trails = 60
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
env = SimpleFleet(inistate, sims)

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
total_steps = 0
for j in range(trails):
    done = False
    score = 0
    step = 0
    trm1s = []
    trm2s = []
    trm3s = []
    trm4s = []
    tru1s = []
    tru2s = []
    tru3s = []
    tru4s = []
    while not done:
        u1t = np.random.uniform(low=-1, high=3)
        u2t = np.random.uniform(low=-1, high=3)
        u3t = np.random.uniform(low=-1, high=3)
        u4t = np.random.uniform(low=-1, high=3)
        # step the game
        s1, s2, s3, s4, done = env.sim([u1t, u2t, u3t, u4t], step)
        cur_state = np.array([s1, s2-s1, s3-s1, s4-s1])
        if not done:
            m1s.append(s1)
            m2s.append(s2)
            m3s.append(s3)
            m4s.append(s4) 
            u1s.append(u1t) 
            u2s.append(u2t) 
            u3s.append(u3t) 
            u4s.append(u4t) 

        total_steps += 1                
        step += 1

# plot one game traj
fig0 = plt.figure(figsize=(8,6))
plt.subplot(1,1,1)
# plt.plot(m1s[3], label='agent1')
plt.plot(m1s, label='agent1')
plt.plot(m2s, label='agent2')
plt.plot(m3s, label='agent3')
plt.plot(m4s, label='agent4')
plt.xlabel("time steps",fontsize=12)
plt.ylabel("position",fontsize=12)
plt.legend(loc='upper right', prop={'size': 8})
plt.title('One trail trajectory during data collecting',fontsize=12,color='black')

fig1 = plt.figure(figsize=(8,6))
plt.subplot(1,1,1)
# plt.plot(m1s[3], label='agent1')
plt.plot(u1s, label='agent1')
plt.plot(u2s, label='agent2')
plt.plot(u3s, label='agent3')
plt.plot(u4s, label='agent4')
plt.xlabel("time steps",fontsize=12)
plt.ylabel("control input",fontsize=12)
plt.legend(loc='upper right', prop={'size': 8})
plt.title('One trail control input during data collecting',fontsize=12,color='black')

state = np.zeros((8,len(m1s)))
state[0,:] = np.squeeze(np.array([m1s]).reshape(1,-1))
state[1,:] = np.squeeze(np.array([m2s]).reshape(1,-1)) - state[0,:]
state[2,:] = np.squeeze(np.array([m3s]).reshape(1,-1)) - state[0,:]
state[3,:] = np.squeeze(np.array([m4s]).reshape(1,-1)) - state[0,:]
state[4,:] = np.squeeze(np.array([u1s]).reshape(1,-1))
state[5,:] = np.squeeze(np.array([u2s]).reshape(1,-1))
state[6,:] = np.squeeze(np.array([u3s]).reshape(1,-1))
state[7,:] = np.squeeze(np.array([u4s]).reshape(1,-1))

filename = 'data/trainingdata.pkl'
joblib.dump(state, filename)
print('Saved data, data dimension: 8 X', trails*sims)
plt.show()
