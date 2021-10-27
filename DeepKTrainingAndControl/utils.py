'''
===================================================================================
Description: Models used for kernel linerization of koopman operator
Project: Data driven control for the multiagent fleet
Author: Wenjian Hao, Purdue University

Version: Sep / 2021
===================================================================================
'''


import torch
import control
import joblib

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchsummary import summary
from LNN import LNN
  
#=====================
# training loop
#=====================
class DKRC_training():
    def __init__(self, data_path, data_name, model_saved_name, training_epoch, dim_states, dim_lifting, dim_control, 
                  learning_rate, decay_rate, keep_train, plot_NN):

      self.data_path = data_path
      self.data_name = data_name
      self.model_saved_name = model_saved_name
      self.training_epoch = training_epoch
      self.dim_states = dim_states
      self.dim_lifting = dim_lifting
      self.dim_control = dim_control
      self.learning_rate = learning_rate
      self.keep_train = keep_train
      self.plot_NN = plot_NN   
      self.decay_rate = decay_rate

    def train_model(self):
      # get the training data
      train_samples, train_label, train_u, valid_samples, valid_label, valid_u = self.data_process(split=True)
      X_lift_table = np.empty(shape=[self.dim_lifting, 0])
      Y_lift_table = np.empty(shape=[self.dim_lifting, 0])
      U_table = np.empty(shape=[self.dim_control, 0])
      XU = []
      training_device = torch.device('cuda:0')
      print('It will be trained on the device: ', training_device)

      # set parameters
      if self.keep_train:
        load_liftNN = self.data_path + self.model_saved_name
        ckpt_liftNN = torch.load(load_liftNN, map_location='cuda:0')
        lifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
        lifting.load_state_dict(ckpt_liftNN['model_lifting'])
        print('Keep Training')
      else:
        lifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
      
      # plot structure
      if self.plot_NN:
          device = torch.device("cuda")
          modellifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting).to(device)
          summary(modellifting, (1,self.dim_states))

      #==================================================
      # choose the training optimizer
      #==================================================
      # optimizer = optim.Adam(lifting.parameters(), lr=1e-4, weight_decay = 0.01)
      # optim_decoder = optim.AdamW(decoder.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
      # optim_decoder = optim.Adamax(decoder.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
      # optim_decoder = optim.Adam(decoder.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
      
      # for converging quickly
      optimizer = optim.Adam(lifting.parameters(), lr=self.learning_rate, weight_decay = self.decay_rate)
      #==================================================
    
      train_loss = []
      valid_loss = []

      # Pass data to the training GPU
      tx, ty, tu, vx, vy, vu = train_samples, train_label, train_u, valid_samples, valid_label, valid_u
      trset = torch.FloatTensor(tx).to(training_device)
      trlab = torch.FloatTensor(ty).to(training_device)
      tru = torch.FloatTensor(tu).to(training_device)
      vlset = torch.FloatTensor(vx).to(training_device)
      vllab = torch.FloatTensor(vy).to(training_device)
      vlu = torch.FloatTensor(vu).to(training_device)
      trset = torch.transpose(trset, 0, 1)
      trlab = torch.transpose(trlab, 0, 1)
      tru = torch.transpose(tru, 0, 1)
      vlset = torch.transpose(vlset, 0, 1)
      vllab = torch.transpose(vllab, 0, 1)
      vlu = torch.transpose(vlu, 0, 1)
      lifting.to(training_device)

      # Training
      lifting.train()
      for i in range(self.training_epoch):
          # lifting xt and x(t+1)
          x_t_lift = lifting(trset)
          x_t1_lift = lifting(trlab)
          
          #==================================
          # Calculating A, B matrices
          #==================================
          # Initialize the matrix stack for A, B matrices calculation
          x_lift_stack = x_t_lift.cpu().detach().numpy()
          x_t1_lift_stack = x_t1_lift.cpu().detach().numpy()
          u_t = tru.cpu().detach().numpy()   # To do: need to be ensured that if we need a contrains for the initial state
          X_lift_table = np.append(X_lift_table, x_lift_stack.T, axis=1)	  # horizontally concatenation NxK
          Y_lift_table = np.append(Y_lift_table, x_t1_lift_stack.T, axis=1)	# horizontally concatenation NxK
          U_table = np.append(U_table, np.matrix(u_t).T, axis=1)
          # A and B, which can be found in paper 
          # "Linear predictors for nonlinear dynamicalsystems: Koopman operator meets model predictive control"
          XU = np.append(X_lift_table,U_table, axis=0)
          G = XU * (XU.T)
          V = Y_lift_table * (XU.T)
          M = V * np.linalg.pinv(G)     
          # get the A,B,C matrix     
          A = M[:, 0:self.dim_lifting]
          B = M[:, self.dim_lifting:(self.dim_lifting+self.dim_control)]
          A_mat = torch.FloatTensor(A).to(training_device)
          B_mat = torch.FloatTensor(B).to(training_device)
          C_mat = torch.matmul(trset.T, torch.linalg.pinv(x_t_lift).T).to(training_device)
          D_mat = torch.zeros(C_mat.shape[0], B_mat.shape[1]).to(training_device)
          AB_mat = torch.cat((A_mat,B_mat),1)
          CD_mat = torch.cat((C_mat, D_mat),1)
          ABCD_mat = torch.cat((AB_mat, CD_mat), 0)
          #==================================
          
          # loss1 is the linear lifting loss
          loss1 = self.linearize_loss(ABCD_mat, x_t_lift, x_t1_lift, trset, tru)
          # loss2 ensures a controllable new lifted linear system
          rank, loss2 = self.controllability_loss(A, B)
          # get the complete loss 
          total_loss = loss1 + loss2

          # update weights at each epoch
          if torch.norm(B_mat) > 0.0005:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
          else:
            total_loss.backward()

          # evaluation
          lifting.eval()
          with torch.set_grad_enabled(False):
            vx_t_lift = lifting(vlset)
            vx_t1_lift = lifting(vllab)
            # validation loss1
            vloss1 = self.linearize_loss(ABCD_mat, vx_t_lift, vx_t1_lift, vlset, vlu)
            # loss3 is the decoder loss
            vtotal_loss = vloss1 + loss2

          # plot the training process
          if i % 10 == 0:
            print('Current epoch is: ', i)
            print('Training loss is: %.9f' % (total_loss))
            print('Validating loss is: %.9f' % (vtotal_loss))
          
          # only saving the model with the lowest loss
          train_loss.append(total_loss.cpu().detach().numpy())
          valid_loss.append(vtotal_loss.cpu().detach().numpy())
          min_loss = min(train_loss)
          if total_loss.cpu().detach().numpy() <= min_loss:   
              if rank == self.dim_lifting:
                  # save lifting and decoder models
                  state = {'model_lifting': lifting.state_dict()}
                  torch.save(state, (self.data_path+self.model_saved_name))

                  # plot the identity matrices
                  # print('A matrix is: ' + '\n', A)
                  # print('B matrix is: ' + '\n', B)
                  # # print('ctrb is: '+ '\n', ctrb)
                  # print('Rank is: '+ '\n', rank)
                  # print('C matrix is: ' + '\n', C)

                  # Save the lifted dynamics
                  filename_a = self.data_path + 'A.pkl'
                  filename_b = self.data_path + 'B.pkl'
                  filename_c = self.data_path + 'C.pkl'
                  joblib.dump(A, filename_a)
                  joblib.dump(B, filename_b) 
                  joblib.dump(C_mat.cpu().detach().numpy(), filename_c)
                  print("Saved min loss model, loss: ", total_loss.cpu().detach().numpy())

      # visualize training process
      plt.plot(train_loss,label='Training loss')
      plt.plot(valid_loss,"*",label='Validation loss')
      plt.title('Training Process', fontsize=23)
      plt.ylabel('Loss', fontsize=20)
      plt.xlabel('Epoch', fontsize=20)
      plt.xticks(fontsize=20)
      plt.yticks(fontsize=20)
      plt.legend(loc='upper right', prop={'size': 20})
      plt.show()

    #=====================
    # Data process
    #=====================
    def data_process(self, split):
      data = joblib.load(self.data_name)
      Z = data[0:(self.dim_states+self.dim_control),:]
      X = np.zeros((Z.shape[0],0))
      Y = np.zeros((Z.shape[0],0))
      X = np.append(X, Z[:,0:-2], 1)  # X_t
      Y = np.append(Y, Z[:,1:-1], 1)  # X_(t+1)

      samples = X[0:(self.dim_states), :]
      label = Y[0:(self.dim_states), :]
      U_control = X[self.dim_states:(self.dim_states+self.dim_control), :]

      # split data
      train_len = int(0.8*len(label[0]))
      totalen = len(X[0])
      train_samples = samples[:, 0:(train_len)]
      train_label = label[:, 0:(train_len)]
      train_u = U_control[:, 0:(train_len)]
      valid_samples = samples[:, train_len:totalen]
      valid_label = label[:, train_len:totalen]
      valid_u = U_control[:, train_len:totalen]
      
      if split:
        return train_samples, train_label, train_u, valid_samples, valid_label, valid_u
      else:
        return samples

    #=====================
    # loss functions
    #=====================
    # def linearize_loss(self, A_mat, B_mat, x_t_lift, x_t1_lift, tru):
    #   prediction = A_mat @ torch.transpose(x_t_lift,0,1) + B_mat @ torch.transpose(tru,0,1)
    #   lossfunc = torch.nn.MSELoss()
    #   loss1 = lossfunc(prediction, torch.transpose(x_t1_lift,0,1))
    #   return loss1

    def linearize_loss(self, ABCD_mat, x_t_lift, x_t1_lift, trset, tru):
      label = torch.cat((x_t1_lift.T, trset.T), 0)
      trainset = torch.cat((x_t_lift.T, tru.T), 0)
      prediction = ABCD_mat @ trainset
      lossfunc = torch.nn.MSELoss()
      loss1 = lossfunc(prediction, label)
      return loss1

    def controllability_loss(self, A, B):
      ctrb = control.ctrb(A, B)
      rank = np.linalg.matrix_rank(ctrb)
      loss2 = self.dim_lifting - rank
      return rank, loss2
