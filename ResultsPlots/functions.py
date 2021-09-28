'''
function file
'''
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
import matplotlib.pyplot as plt

def load_data(filepath):

  history = joblib.load(filepath)
  history = history.T

  costheta = history[:,0]
  sintheta = history[:,1]
  theta_dot = history[:,2]
  action = history[:,3]

  goal_costheta = 0
  goal_sintheta = 0
  goal_theta_dot = 0

  costheta -= goal_costheta
  sintheta -= goal_sintheta
  theta_dot -= goal_theta_dot

  Z = np.array([costheta, sintheta, theta_dot, action])
  X,Y = fix_data(Z)

  return X, Y

def fix_data(Z):
  X = np.zeros((Z.shape[0],0))
  Y = np.zeros((Z.shape[0],0))
  X = np.append(X, Z[:,0:-2], 1)
  Y = np.append(Y, Z[:,1:-1], 1)
  return X,Y

def data_process(data_name, NUM_PHISICAL_STATES):
  filename_in = data_name
  X, Y = load_data(filename_in)

  samples = X[0:(NUM_PHISICAL_STATES), :]
  label = Y[0:(NUM_PHISICAL_STATES), :]
  U_control = X[3, :]
  U_control = U_control.reshape(1,np.size(U_control))

  # split data
  train_len = int(0.8*len(label[0]))
  totalen = len(X[0])
  train_samples = samples[:, 0:(train_len)]
  train_label = label[:, 0:(train_len)]
  train_u = U_control[:, 0:(train_len)]
  valid_samples = samples[:, train_len:totalen]
  valid_label = label[:, train_len:totalen]
  valid_u = U_control[:, train_len:totalen]
  return train_samples, train_label, train_u, valid_samples, valid_label, valid_u

def train_model(train_samples, train_label, train_u, valid_samples, valid_label, valid_u, NUM_PHISICAL_STATES, NUM_LIFT_DIM, NUM_CONTROL, data_path, filename, num_epoch):
  X_table = np.empty(shape=[NUM_PHISICAL_STATES, 0])
  X_lift11 = np.empty(shape=[NUM_LIFT_DIM, 0])
  Y_lift = np.empty(shape=[NUM_LIFT_DIM, 0])
  U = np.empty(shape=[NUM_CONTROL, 0])
  XU = []
  training_device = torch.device('cuda:0')
  print('WILL USE ', training_device)

  # set parameters
  
  lambda1 = 0
  lambda2 = 0
  lambda3 = 0
  model_PSI = PSI_NN()
  model_A = A_NN()
  model_B = B_NN()
  optimizer = optim.Adam(
                          model_PSI.parameters(), lr=1e-3)
                          # {'params': model_B.parameters()},
                          # {'params': model_A.parameters()}], lr=1e-3
                          #                                             )                  
  train_loss = []
  valid_loss = []
  for i in range(num_epoch):
      print('Current epoch is: ', i)
      model_PSI.to(training_device)
      # model_A.to(training_device)
      # model_B.to(training_device)
      
      model_PSI.train()
      # model_A.train()
      # model_B.train()
      trainloss = 0
      loss = 0
      loss2 = 0

      for j in range (len(train_label[0])):
        x, y, u= train_samples, train_label, train_u
        trainingset = x[:, j]
        labelset = y[:, j]
        ulabelset = u[:,j]
        trset = torch.from_numpy(trainingset).float().to(training_device)
        labset = torch.from_numpy(labelset).float().to(training_device)
        ulabset = torch.from_numpy(ulabelset).float().to(training_device)
        
        # A * PSI(Xt)
        # psi_0 = np.matrix([np.pi,8])
        psi_0 = np.matrix([1,0,0])
        psi_0 = torch.from_numpy(psi_0).float().to(training_device)
        psi_0 = model_PSI(psi_0)

        x_lift = model_PSI(trset) - psi_0
        # A_x_lift = model_A(x_lift)

        # PSI(Xt+1)
        x_next_lift = model_PSI(labset) - psi_0
        # x_next_lift = labset
        loss = torch.norm((x_next_lift - (x_next_lift * torch.pinverse(x_lift)) * x_lift), p=2)
        trainloss += loss

        x_lift1 = x_lift.cpu().detach().numpy()
        x_next_lift1 = x_next_lift.cpu().detach().numpy()
        U0 = np.array([0])
        B_U1 = ulabelset - U0

        X_lift11 = np.append(X_lift11, x_lift1.T, axis=1)	# horizontally concatenation NxK
        Y_lift = np.append(Y_lift, x_next_lift1.T, axis=1)	# horizontally concatenation NxK
        # pdb.set_trace()
        U = np.append(U, np.matrix(B_U1).T, axis=1)

      # A and B
      XU = np.append(X_lift11,U, axis=0)	# vertically concatenation (N+m)xK
      # AB = Y_lift.dot(np.linalg.pinv(XU))	# this equation is a function of the size of training samples K, not suggested by Mezic
      AB = Y_lift*XU.T*np.linalg.pinv(XU*XU.T)
      A = AB[:, 0:NUM_LIFT_DIM]
      B = AB[:, NUM_LIFT_DIM: NUM_LIFT_DIM+NUM_CONTROL]
      ctrb = control.ctrb(A, B)
      rank = np.linalg.matrix_rank(ctrb)
      loss2 = NUM_LIFT_DIM - rank

      losses = trainloss/len(train_label[0]) + loss2
      
      # update weights each epoch
      optimizer.zero_grad()
      (loss+loss2).backward()
      optimizer.step()

      if i % 1 == 0:
          print('train loss is: %.9f' % losses)
          train_loss.append(losses)

      # evaluation
      model_PSI.eval()
      model_A.eval()
      model_B.eval()
      validloss = 0
      vloss = 0

      with torch.set_grad_enabled(False):
        for k in range(len(valid_label[1])):
            vx, vy, vu = valid_samples, valid_label, valid_u
            vaset = vx[:, k]
            vlset = vy[:, k]
            vuset = vu[:, k]
            vaset = torch.from_numpy(vaset).float().to(training_device)
            vlset = torch.from_numpy(vlset).float().to(training_device) 
            vuset = torch.from_numpy(vuset).float().to(training_device)
            # psi_0 = np.matrix([np.pi,8])
            psi_0 = np.matrix([1,0,0])
            psi_0 = torch.from_numpy(psi_0).float().to(training_device)
            psi_0 = model_PSI(psi_0)
            vout = model_PSI(vaset) - psi_0
            vlabel = model_PSI(vlset) - psi_0
            vloss = torch.norm((vlabel - (vlabel * torch.pinverse(vout)) * vout), p=2)
            validloss += vloss

      validloss += loss2
      
      if i % 1 == 0:
          print('Valid Loss is: %.9f' % (validloss/(k+1)))
          valid_loss.append(validloss/(k+1))
          print('Rank is: ', np.linalg.matrix_rank(ctrb))
          

      if i % 10 == 0 and i > 0 and rank == NUM_LIFT_DIM:
          # save final model
          state = {
                    'model_PSI': model_PSI.state_dict(),
                    # 'model_A': model_A.state_dict(),
                    # 'model_B': model_B.state_dict()
                  }
          torch.save(state, filename)
          print('A matrix is: ' + '\n', A)
          print('B matrix is: ' + '\n', B)
          
          # print('ctrb is: '+ '\n', ctrb)
          print('Rank is: '+ '\n', np.linalg.matrix_rank(ctrb))
          filename_a = data_path + 'A.pkl'
          filename_b = data_path + 'B.pkl'
          joblib.dump(A, filename_a)
          joblib.dump(B, filename_b) 

          print("HAVE FINISHED TRAINING, PLEASE CHECK THE MODEL: " + filename)

  # visualize training process
  plt.plot(train_loss)
  plt.plot(valid_loss)
  plt.title('Training Process')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Valid'], loc='upper right')
  plt.show()

def computer_a_b_c(NUM_PHISICAL_STATES, NUM_LIFT_DIM):
  # load models
  load_name = 'data/6_6.pth'
  checkpoint = torch.load(load_name)
  model = PSI_NN()
  # modela = A_NN()
  # modelb = B_NN()
  model.load_state_dict(checkpoint['model_PSI'])
  # C matrix
  data_path = 'data/'
  filename_in = data_path + '/data.pkl'
  X, Y = load_data(filename_in)
  x_ori = X[0:NUM_PHISICAL_STATES, :]
  x_lift = np.zeros((NUM_LIFT_DIM,len(X[1])))
  for i in range(len(X[1])):
      psi_0 = np.matrix([1,0,0])
      psi_0 = torch.from_numpy(psi_0).float()
      psi_0 = model(psi_0)
      x = x_ori[:,i]
      input_tensor = torch.from_numpy(x.T).float()
      cur_state_lifted = model(input_tensor)# - psi_0
      cur_state_lifted = np.matrix(cur_state_lifted.cpu().detach().numpy()) 
      x_lift[:,i] = cur_state_lifted

      # x = x_ori[:,i]
      # input_tensor = torch.from_numpy(x.T).float()
      # cur_state_lifted = model(input_tensor).cpu().detach().numpy()
      # cur_state_lifted = np.matrix(cur_state_lifted.T) 
      # x_lift[:,i] = cur_state_lifted

  C_D = np.matrix(x_ori) * np.linalg.pinv(np.matrix(x_lift))
  print('C matrix is: ' + '\n', C_D)
  filename_c1 = data_path + 'C.pkl'
  joblib.dump(C_D, filename_c1)
  print('saved A, B and C matrices')
