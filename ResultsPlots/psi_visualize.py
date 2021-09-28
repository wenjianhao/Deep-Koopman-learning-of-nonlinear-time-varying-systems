import math
import numpy as np
import torch
import pickle
import scipy
import control

import torch.optim as optim
import torch.nn as nn

from models import PSI_NN, A_NN, B_NN
import joblib
from functions import data_process
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#=====================================================
# loading model and data
#=====================================================
load_name = 'data/withoutu/6_6.pth'
checkpoint = torch.load(load_name)
model = PSI_NN()
model.load_state_dict(checkpoint['model_PSI'])
data_name = 'data/withoutu/data.pkl'
NUM_PHISICAL_STATES = 3
NUM_SAMPLES = 800
# load data
d1, s, train_u, valid_samples, valid_label, valid_u = data_process(data_name, NUM_PHISICAL_STATES)

# #--------------
# xax = []
# yax = []
# for i in range(-20,20):
#     xax.append(np.sin(2*np.pi*i/20))
#     yax.append(np.cos(2*np.pi*i/20))
# plt.plot(xax, yax) 
# plt.show()

x = []
y = []
u = []
energy = []
psival = []
costhe = []
sinthe = []
G = 0
A = 0
for i in range(NUM_SAMPLES):
    # transform theta to [0,pi],[-pi,0]
    if s[1,i] < 0 and s[0,i] < 0:
        realtheta = -np.arccos(s[0,i])
    if s[1,i] < 0 and s[0,i] > 0:
        realtheta = np.arcsin(s[1,i])
    if s[1,i] > 0 and s[0,i] > 0:
        realtheta = np.arcsin(s[1,i])
    if s[1,i] > 0 and s[0,i] < 0:
        realtheta = np.arccos(s[0,i])
        
    if s[1,i] == 0 and s[0,i] == 1:
        realtheta = 0
    if s[1,i] == 0 and s[0,i] == -1:
        realtheta = np.pi
    if s[1,i] == 1 and s[0,i] == 0:
        realtheta = np.pi/2
    if s[1,i] == -1 and s[0,i] == 0:
        realtheta = -np.pi/2
    point = s[:,i].T
    x.append(realtheta)
    y.append(s[2,i])
    u.append(train_u[0,i])
    costhe.append(s[0,i])
    sinthe.append(s[1,i])
    ge = .5*s[2,i]*s[2,i] + s[0,i] + train_u[0,i]
    energy.append(ge)
    input_tensor = torch.from_numpy(point).float()
    psi = model(input_tensor).cpu().detach().numpy()
    psival.append(psi)

kx = s[:,0:-2]
ky = s[:,1:-1]
for k in range(NUM_SAMPLES - 1):
    inkx = torch.from_numpy(kx[:,k].T).float()
    inky = torch.from_numpy(ky[:,k].T).float()
    psix = model(inkx).cpu().detach().numpy()
    psiy = model(inky).cpu().detach().numpy()
    G += np.matrix(psix).T * np.matrix(psix)
    A += np.matrix(psix).T * np.matrix(psiy)

G = G/(NUM_SAMPLES-1)
A = A/(NUM_SAMPLES-1)
x = np.array(x)
y = np.array(y)
costhe = np.array(costhe)
sinthe = np.array(sinthe)
u = np.array(u)
energy = np.array(energy)
psival = np.matrix(psival).T

#===================================================================
# plot eigenvalues
#===================================================================
Koopman = psival[:,1:-1] * np.linalg.pinv(psival[:,0:-2])
# Koopman = np.linalg.pinv(G) * A
evalue, evector = np.linalg.eig(Koopman)
print('eigenvalues are: ' + '\n', evalue)
print('eigenvectors are:' + '\n', evector)

fig0 = plt.figure(figsize=(6,6))
plt.subplot(1,1,1)
for i in range(len(evalue)):
    plt.scatter([0,evalue[i].real],[0,evalue[i].imag],c='b',label='python')
limit=1.25#np.max(np.ceil(np.absolute(evalue))) # set limits for axi=
plt.xlim((-.5,limit))
plt.ylim((-limit,limit))
plt.ylabel('Imaginary',fontsize=20)
plt.xlabel('Real',fontsize=20)
plt.title('Pendulum EIGENVALUES OF KOOPMAN OPERATOR',fontsize=25)

fig00 = plt.figure(figsize=(6,6))
plt.subplot(1,1,1)
plt.scatter(x, y ,c=energy, cmap='jet')
im00 = plt.scatter(x, y ,c=energy, cmap='jet')
plt.xlabel('theta',fontsize=20)
plt.ylabel('theta dot',fontsize=20)
# plt.title('Pendulum 2D plot of theta and theta dot',fontsize=25)
fig00.colorbar(im00)
# plt.subplot(1,2,2)
# plt.scatter(np.array(psival[2,:]).reshape(NUM_SAMPLES), np.array(psival[4,:]).reshape(NUM_SAMPLES) ,c=energy, cmap='jet')
# im01 = plt.scatter(np.array(psival[2,:]).reshape(NUM_SAMPLES), np.array(psival[4,:]).reshape(NUM_SAMPLES) ,c=energy, cmap='jet')
# plt.xlabel('psi1',fontsize=20)
# plt.ylabel('psi2',fontsize=20)
# plt.title('Pendulum 2D plot of psi1 and psi2',fontsize=25)
# fig00.colorbar(im01)
fig000 = plt.figure(figsize=(6,6))
plt.subplot(1,1,1)
plt.scatter((costhe*sinthe+y), energy, c=energy, cmap='jet')
im000 = plt.scatter((costhe*sinthe+y), energy, c=energy, cmap='jet')
plt.xlabel('sum of cos,sin and theta dot',fontsize=20)
plt.ylabel('energy',fontsize=20)
# plt.title('Pendulum 2D plot of theta and theta dot',fontsize=25)
fig000.colorbar(im000)

fig200 = plt.figure(figsize=(12,12))
fig200.set_facecolor('white')
ax = fig200.add_subplot(2,2,1,projection='3d')
ax.scatter(x, y, energy, cmap=plt.cm.Spectral_r)
ax.set_zlabel('energy',fontsize=20)
ax.set_ylabel('theta_dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)
ax.set_title('3d plot of energy,theta and theta dot',fontsize=25)

ax = fig200.add_subplot(2,2,2,projection='3d')
ax.scatter(costhe, y, energy, cmap=plt.cm.Spectral_r)
ax.set_zlabel('energy',fontsize=20)
ax.set_ylabel('theta_dot',fontsize=20)
ax.set_xlabel('cos',fontsize=20)
ax.set_title('3d plot of energy,theta and theta dot',fontsize=25)

ax = fig200.add_subplot(2,2,3,projection='3d')
ax.scatter(sinthe, y, energy, cmap=plt.cm.Spectral_r)
ax.set_zlabel('energy',fontsize=20)
ax.set_ylabel('theta_dot',fontsize=20)
ax.set_xlabel('sin',fontsize=20)
ax.set_title('3d plot of energy,theta and theta dot',fontsize=25)

ax = fig200.add_subplot(2,2,4,projection='3d')
ax.scatter(np.array(psival[5,:]).reshape(NUM_SAMPLES), np.array(psival[6,:]).reshape(NUM_SAMPLES), energy, c=energy)
ax.set_zlabel('energy',fontsize=20)
ax.set_ylabel('psi4',fontsize=20)
ax.set_xlabel('psi5',fontsize=20)
ax.set_title('3d plot of energy,psi1 and psi2',fontsize=25)

fig2 = plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.scatter(costhe, energy, c=energy, cmap='jet')
im020 = plt.scatter(costhe, energy, c=energy, cmap='jet')
plt.xlabel('cos(theta)',fontsize=20)
plt.ylabel('energy',fontsize=20)
# plt.title('Pendulum 2D plot of theta dot and energy',fontsize=25)
fig2.colorbar(im020)

plt.subplot(2,2,2)
plt.scatter(sinthe, energy, c=energy, cmap='jet')
im021 = plt.scatter(sinthe, energy, c=energy, cmap='jet')
plt.xlabel('sin(theta)',fontsize=20)
plt.ylabel('energy',fontsize=20)
# plt.title('Pendulum 2D plot of theta dot and energy',fontsize=25)
fig2.colorbar(im021)

plt.subplot(2,2,3)
plt.scatter(y, energy, c=energy, cmap='jet')
im02 = plt.scatter(y, energy, c=energy, cmap='jet')
plt.xlabel('theta dot',fontsize=20)
plt.ylabel('energy',fontsize=20)
# plt.title('Pendulum 2D plot of theta dot and energy',fontsize=25)
fig2.colorbar(im02)

plt.subplot(2,2,4)
plt.scatter(np.array(psival[6,:]).reshape(NUM_SAMPLES), energy, c=energy, cmap='jet')
im03 = plt.scatter(np.array(psival[6,:]).reshape(NUM_SAMPLES), energy, c=energy, cmap='jet')
plt.xlabel('Dominant eigenfunction',fontsize=20)
plt.ylabel('energy',fontsize=20)
# plt.title('Pendulum 2D plot of psi5 and energy',fontsize=25)
fig2.colorbar(im03)

# fig20 = plt.figure(figsize=(12,6))
# fig20.set_facecolor('white')
# ax = fig20.add_subplot(1,2,1,projection='3d')
# ax.scatter(costhe, sinthe, y, c=energy)
# im1 = ax.scatter(costhe, sinthe, y, c=energy)
# ax.set_zlabel('theta dot',fontsize=20)
# ax.set_ylabel('sin',fontsize=20)
# ax.set_xlabel('cos',fontsize=20)
# # ax.set_title('3d plot of sin, cos and theta dot',fontsize=25)
# fig20.colorbar(im1)
# ax = fig20.add_subplot(1,2,2,projection='3d')
# ax.scatter(np.array(psival[4,:]).reshape(NUM_SAMPLES), np.array(psival[5,:]).reshape(NUM_SAMPLES), np.array(psival[6,:]).reshape(NUM_SAMPLES), c=energy)
# im2 = ax.scatter(np.array(psival[4,:]).reshape(NUM_SAMPLES), np.array(psival[5,:]).reshape(NUM_SAMPLES), np.array(psival[6,:]).reshape(NUM_SAMPLES), c=energy)
# ax.set_zlabel('psi3',fontsize=20)
# ax.set_ylabel('psi2',fontsize=20)
# ax.set_xlabel('psi1',fontsize=20)
# # ax.set_title('3d plot of psi1, psi2 and psi3',fontsize=25)
# fig20.colorbar(im2)
fig2000 = plt.figure(figsize=(12,24))
ax = fig2000.add_subplot(4,2,1,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[0,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi1',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)

ax = fig2000.add_subplot(4,2,2,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[1,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi2',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)

ax = fig2000.add_subplot(4,2,3,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[2,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi3',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)

ax = fig2000.add_subplot(4,2,4,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[3,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi4',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)

ax = fig2000.add_subplot(4,2,5,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[4,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi5',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)

ax = fig2000.add_subplot(4,2,6,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[5,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi6',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)

ax = fig2000.add_subplot(4,2,7,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[6,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi7',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)

ax = fig2000.add_subplot(4,2,8,projection='3d')
ax.plot_trisurf(x,y, np.array(psival[7,:]).reshape(NUM_SAMPLES),cmap=plt.cm.Spectral)
ax.set_zlabel('psi8',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)


#=====================================================================================================================
# plot psi eigenvectors corresponding to theta, theta dot
#=====================================================================================================================
psival = psival.T
psi1 = psival * evector[:,0]
psi2 = psival * evector[:,1]
psi3 = psival * evector[:,2]
psi4 = psival * evector[:,3]
psi5 = psival * evector[:,4]
psi6 = psival * evector[:,5]
psi7 = psival * evector[:,6]
psi8 = psival * evector[:,7]
# shape transformation
psif1 = []
for ele in psi1:
    psif1.append(ele)
psif1 = np.array(psif1).reshape(len(x),)
psif2 = []
for ele in psi2:
    psif2.append(ele)
psif2 = np.array(psif2).reshape(len(x),)
psif3 = []
for ele in psi3:
    psif3.append(ele)
psif3 = np.array(psif3).reshape(len(x),)
psif4 = []
for ele in psi4:
    psif4.append(ele)
psif4 = np.array(psif4).reshape(len(x),)
psif5 = []
for ele in psi5:
    psif5.append(ele)
psif5 = np.array(psif5).reshape(len(x),)
psif6 = []
for ele in psi6:
    psif6.append(ele)
psif6 = np.array(psif6).reshape(len(x),)
psif7 = []
for ele in psi7:
    psif7.append(ele)
psif7 = np.array(psif7).reshape(len(x),)
psif8 = []
for ele in psi8:
    psif8.append(ele)
psif8 = np.array(psif8).reshape(len(x),)

# #=====================================================
# # plot setting (real part of eigenvectors)
# #=====================================================
# fig = plt.figure(figsize=(12,24))
# fig.set_facecolor('white')

# # psi1
# ax = fig.add_subplot(4, 2, 1, projection='3d')
# ax.plot_trisurf(x, y, psif1.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('1st & 2nd',fontsize=20)

# # # psi2
# # ax = fig.add_subplot(4, 2, 2, projection='3d')
# # ax.plot_trisurf(x, y, psif2.real, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi2')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi2 fuction with real number part')

# # psi3
# ax = fig.add_subplot(4, 2, 3, projection='3d')
# ax.plot_trisurf(x, y, psif3.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('3rd & 4th',fontsize=20)

# # # psi4
# # ax = fig.add_subplot(4, 2, 4, projection='3d')
# # ax.plot_trisurf(x, y, psif4.real, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi4')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi4 fuction with real number part')

# # psi5
# ax = fig.add_subplot(4, 2, 5, projection='3d')
# ax.plot_trisurf(x, y, psif5.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('5th & 6th',fontsize=20)

# # # psi6
# # ax = fig.add_subplot(4, 2, 6, projection='3d')
# # ax.plot_trisurf(x, y, psif6.real, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi6')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi6 fuction with real number part')

# # psi7
# ax = fig.add_subplot(4, 2, 7, projection='3d')
# ax.plot_trisurf(x, y, psif7.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('7th & 8th',fontsize=20)

# # # psi8
# # ax = fig.add_subplot(4, 2, 8, projection='3d')
# # ax.plot_trisurf(x, y, psif8.real, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi8')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi8 fuction with real number part')

# #=====================================================
# # plot setting (imaginary part of eigenvectors)
# #=====================================================
# # fig1 = plt.figure(figsize=(12,12))
# # fig1.set_facecolor('white')
# # psi1
# ax = fig.add_subplot(4, 2, 2, projection='3d')
# ax.plot_trisurf(x, y, psif1.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('1st & -2nd',fontsize=20)

# # # psi2
# # ax = fig1.add_subplot(4, 2, 2, projection='3d')
# # ax.plot_trisurf(x, y, psif2.imag, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi2')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi2 fuction with imaginary number part')

# # psi3
# ax = fig.add_subplot(4, 2, 4, projection='3d')
# ax.plot_trisurf(x, y, psif3.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('3rd & -4th',fontsize=20)

# # # psi4
# # ax = fig1.add_subplot(4, 2, 4, projection='3d')
# # ax.plot_trisurf(x, y, psif4.imag, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi4')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi4 fuction with imaginary number part')

# # psi5
# ax = fig.add_subplot(4, 2, 6, projection='3d')
# ax.plot_trisurf(x, y, psif5.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('5th & -6th',fontsize=20)

# # # psi6
# # ax = fig1.add_subplot(4, 2, 6, projection='3d')
# # ax.plot_trisurf(x, y, psif6.imag, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi6')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi6 fuction with imaginary number part')

# # psi7
# ax = fig.add_subplot(4, 2, 8, projection='3d')
# ax.plot_trisurf(x, y, psif7.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('7th & -8th',fontsize=20)

# # # psi8
# # ax = fig1.add_subplot(4, 2, 8, projection='3d')
# # ax.plot_trisurf(x, y, psif8.imag, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi8')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi8 fuction with imaginary number part')

# plt.show()







# #psi2
# ax1 = fig.add_subplot(2, 2, 2, projection='3d')
# ax1.plot_trisurf(x, y, psi_output2, cmap=plt.cm.Spectral)
# ax1.set_zlabel('psi_2')
# ax1.set_ylabel('theta_dot')
# ax1.set_xlabel('theta')
# ax1.set_title('psi2')
# #psi3
# ax2 = fig.add_subplot(2, 2, 3, projection='3d')
# ax2.plot_trisurf(x, y, psi_output3, cmap=plt.cm.Spectral)
# ax2.set_zlabel('psi_3')
# ax2.set_ylabel('theta_dot')
# ax2.set_xlabel('theta')
# ax2.set_title('psi3')
# plt.tight_layout()

# #-------------------------------------------------------------------------------------
# # squeeze psi to one figure
# #-------------------------------------------------------------------------------------
# fig1 = plt.figure(figsize=plt.figaspect(0.5))
# fig1.set_facecolor('white')
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(psi_output1, psi_output2, psi_output3, cmap=plt.cm.Spectral)
# ax.set_xlabel('psi1')
# ax.set_ylabel('psi2')
# ax.set_zlabel('psi3')
# ax.set_title('3 psi planes')

# #-----------------------------------------------------
# # plot intersection of psi function
# #-----------------------------------------------------
# fig2 = plt.figure(figsize=plt.figaspect(0.5))
# fig2.set_facecolor('white')
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(x, y, psi_output1, cmap=plt.cm.Spectral)
# ax.plot_trisurf(x, y, psi_output2, cmap=plt.cm.Spectral)
# ax.plot_trisurf(x, y, psi_output3, cmap=plt.cm.Spectral)
# ax.set_xlabel('theta')
# ax.set_ylabel('theta_dot')
# ax.set_zlabel('psi function')
# ax.set_title('intersection')

# #----------------------------------------------------
# # plot energy
# #----------------------------------------------------
# fig0 = plt.figure(figsize=plt.figaspect(0.5))
# fig0.set_facecolor('white')
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(x, y, energy, cmap=plt.cm.Spectral)
# ax.set_xlabel('theta')
# ax.set_ylabel('control')
# ax.set_zlabel('energy')
# ax.set_title('energy plot')
# plt.show()

#=====================================================
# plot setting (real part of eigenvectors)
#=====================================================
fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')

# psi1
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_trisurf(x, y, psif1.real, cmap=plt.cm.Spectral)
ax.set_zlabel('eigenfunc',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)
ax.set_title('1st & 2nd',fontsize=20)

# # psi2
# ax = fig.add_subplot(4, 2, 2, projection='3d')
# ax.plot_trisurf(x, y, psif2.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('psi2')
# ax.set_ylabel('theta_dot')
# ax.set_xlabel('theta')
# ax.set_title('psi2 fuction with real number part')

# psi3
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_trisurf(x, y, psif3.real, cmap=plt.cm.Spectral)
ax.set_zlabel('eigenfunc',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)
ax.set_title('3rd & 4th',fontsize=20)

# # psi4
# ax = fig.add_subplot(4, 2, 4, projection='3d')
# ax.plot_trisurf(x, y, psif4.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('psi4')
# ax.set_ylabel('theta_dot')
# ax.set_xlabel('theta')
# ax.set_title('psi4 fuction with real number part')

# # psi5
# ax = fig.add_subplot(2, 2, 5, projection='3d')
# ax.plot_trisurf(x, y, psif5.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('5th & 6th',fontsize=20)

# # psi6
# ax = fig.add_subplot(4, 2, 6, projection='3d')
# ax.plot_trisurf(x, y, psif6.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('psi6')
# ax.set_ylabel('theta_dot')
# ax.set_xlabel('theta')
# ax.set_title('psi6 fuction with real number part')

# # psi7
# ax = fig.add_subplot(2, 2, 7, projection='3d')
# ax.plot_trisurf(x, y, psif7.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('7th & 8th',fontsize=20)

# # psi8
# ax = fig.add_subplot(4, 2, 8, projection='3d')
# ax.plot_trisurf(x, y, psif8.real, cmap=plt.cm.Spectral)
# ax.set_zlabel('psi8')
# ax.set_ylabel('theta_dot')
# ax.set_xlabel('theta')
# ax.set_title('psi8 fuction with real number part')

#=====================================================
# plot setting (imaginary part of eigenvectors)
#=====================================================
# fig1 = plt.figure(figsize=(12,12))
# fig1.set_facecolor('white')
# psi1
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_trisurf(x, y, psif1.imag, cmap=plt.cm.Spectral)
ax.set_zlabel('eigenfunc',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)
ax.set_title('1st & -2nd',fontsize=20)

# # psi2
# ax = fig1.add_subplot(4, 2, 2, projection='3d')
# ax.plot_trisurf(x, y, psif2.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('psi2')
# ax.set_ylabel('theta_dot')
# ax.set_xlabel('theta')
# ax.set_title('psi2 fuction with imaginary number part')

# psi3
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_trisurf(x, y, psif3.imag, cmap=plt.cm.Spectral)
ax.set_zlabel('eigenfunc',fontsize=20)
ax.set_ylabel('theta dot',fontsize=20)
ax.set_xlabel('theta',fontsize=20)
ax.set_title('3rd & -4th',fontsize=20)

# # psi4
# ax = fig1.add_subplot(4, 2, 4, projection='3d')
# ax.plot_trisurf(x, y, psif4.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('psi4')
# ax.set_ylabel('theta_dot')
# ax.set_xlabel('theta')
# ax.set_title('psi4 fuction with imaginary number part')

# # psi5
# ax = fig.add_subplot(4, 2, 6, projection='3d')
# ax.plot_trisurf(x, y, psif5.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('5th & -6th',fontsize=20)

# # # psi6
# # ax = fig1.add_subplot(4, 2, 6, projection='3d')
# # ax.plot_trisurf(x, y, psif6.imag, cmap=plt.cm.Spectral)
# # ax.set_zlabel('psi6')
# # ax.set_ylabel('theta_dot')
# # ax.set_xlabel('theta')
# # ax.set_title('psi6 fuction with imaginary number part')

# # psi7
# ax = fig.add_subplot(4, 2, 8, projection='3d')
# ax.plot_trisurf(x, y, psif7.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('eigenfunc',fontsize=20)
# ax.set_ylabel('theta dot',fontsize=20)
# ax.set_xlabel('theta',fontsize=20)
# ax.set_title('7th & -8th',fontsize=20)

# # psi8
# ax = fig1.add_subplot(4, 2, 8, projection='3d')
# ax.plot_trisurf(x, y, psif8.imag, cmap=plt.cm.Spectral)
# ax.set_zlabel('psi8')
# ax.set_ylabel('theta_dot')
# ax.set_xlabel('theta')
# ax.set_title('psi8 fuction with imaginary number part')

plt.show()