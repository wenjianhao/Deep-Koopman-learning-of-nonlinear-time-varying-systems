'''
Description: Paper codes of 'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." 
                            Automatica 159 (2024): 111372.'

Author: Wenjian Hao, Purdue University

This is a file containing all the functions.

Start at: Sep 2021
Last Revision: Jan 2022
'''

import torch
import config
import control

import numpy as np
import torch.optim as optim

from LNN import LNN
from torchsummary import summary


#=====================
# training loop
#=====================
class DKTV_training(object):
    
    def __init__(self, config: dict):
      self.plot_NN = config.flags['plot_NN']
      self.keep_train = config.flags['keep_train']

      self.results_path = config.files_dir['results_path']
      self.data_name = config.files_dir['data_name']
      self.model_saved_name = config.files_dir['model_saved_name'] 
      
      self.dim_states = config.dimensions['dim_states']
      self.dim_lifting = config.dimensions['dim_lifting']
      
      self.eps = config.training_parameters['eps']
      self.decay_rate = config.training_parameters['decay_rate']
      self.learning_rate = config.training_parameters['learning_rate']
      self.training_epoch = config.training_parameters['training_epoch']

      # training devices, uncomment for gpu
      # print("Torch version: {}".format(torch.__version__))
      # cuda = torch.cuda.is_available()
      # self.training_device = torch.device("cuda" if cuda else "cpu")
      # print("Training device {}".format(self.training_device))
      self.training_device = torch.device("cpu")

    def pretrain_model(self, tx, ty):
      # some flags
      # keep training?
      if self.keep_train:
        load_liftNN = self.results_path + self.model_saved_name
        ckpt_liftNN = torch.load(load_liftNN, map_location=self.training_device)
        lifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
        lifting.load_state_dict(ckpt_liftNN['model_lifting'])
        print('Keep Training')
      else:
        lifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
      # plot DNNs structure
      if self.plot_NN:
          device = self.training_device
          modellifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting).to(device)
          summary(modellifting, (1,self.dim_states))

      # Training
      train_loss = []
      # get the training data and pass it to the training GPU
      trset = torch.t(torch.FloatTensor(tx).to(self.training_device))
      trlab = torch.t(torch.FloatTensor(ty).to(self.training_device))
      lifting.to(self.training_device)
      # choose the training optimizer
      optimizer = optim.Adam(lifting.parameters(), lr=self.learning_rate, weight_decay = self.decay_rate)
      lifting.train()
      for i in range(self.training_epoch):
          # lifting xt and x(t+1)
          x_t_lift = lifting(trset)
          x_t1_lift = lifting(trlab)
          # get the A, C matrices
          A = torch.t(x_t1_lift)@torch.pinverse(torch.t(x_t_lift))
          C = trset.T@torch.pinverse(x_t_lift.T)
          # get the K matrix in paper
          AC_mat = torch.cat((A, C), 0)
          AC_mat = np.matrix(AC_mat.cpu().detach().numpy())
          AC_mat = torch.FloatTensor(AC_mat).to(self.training_device)
          # loss1 is the linear lifting loss
          loss1, p1 = self.linearize_loss(AC_mat, x_t_lift, x_t1_lift, trset)
          total_loss = loss1
          # gradient descent
          if total_loss > self.eps:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
          # only saving the model with the lowest loss
          train_loss.append(total_loss.cpu().detach().numpy())
          min_loss = min(train_loss)
          if total_loss.cpu().detach().numpy() <= min_loss:   
              state = {'model_lifting': lifting.state_dict()}
              torch.save(state, (self.results_path+'nnbasis/'+str(0)+self.model_saved_name))
              print("Saved min loss model, loss: ", total_loss.cpu().detach().numpy())
          # break if the esitimation accuracy is satisfied
          if total_loss <= self.eps:
              state = {'model_lifting': lifting.state_dict()}
              torch.save(state, (self.results_path+'nnbasis/'+str(0)+self.model_saved_name))
              break
      return A.cpu().detach().numpy(), C.cpu().detach().numpy()

    def DKTV(self, Astk, tx, ty, batchsize, nd):
      # load pretrained DNN
      load_liftNN = self.results_path +'nnbasis/'+str(nd-1) + self.model_saved_name
      ckpt_liftNN = torch.load(load_liftNN, map_location=self.training_device)
      lifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
      lifting.load_state_dict(ckpt_liftNN['model_lifting'])
      print('Finish loading the pretrained model')
      # plot DNN structure or not
      if self.plot_NN:
          device = torch.device("cuda")
          modellifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting).to(device)
          summary(modellifting, (1,self.dim_states))
      # Training
      # pass data to the training device
      trset = torch.t(torch.FloatTensor(tx).to(self.training_device))
      trlab = torch.t(torch.FloatTensor(ty).to(self.training_device))
      lifting.to(self.training_device)
      fixlifting = lifting
      # split the data by time
      xold = trset[0:(trset.size(dim=0)-batchsize), :]
      yold = trlab[0:(trlab.size(dim=0)-batchsize), :]
      xnew = trset[(trset.size(dim=0)-batchsize-1):, :]
      ynew = trlab[(trset.size(dim=0)-batchsize-1):, :]
      # previous solution
      x_t_old = fixlifting(xold)
      x_t1_old = fixlifting(yold)
      XU = torch.t(x_t_old)
      G = XU@torch.t(XU)
      Ap = torch.FloatTensor(Astk[:,:,(nd-1)]).to(self.training_device)
      # choose the training optimizer
      optimizer = optim.Adam(lifting.parameters(), lr=self.learning_rate, weight_decay = self.decay_rate)
      train_loss = []
      lifting.train()
      for i in range(self.training_epoch):
          # lifting xt and x(t+1)
          x_t_lift = lifting(xnew)
          x_t1_lift = lifting(ynew)          
          try:
            chinew = torch.t(x_t_lift)
            lambdatau = torch.inverse(torch.eye(chinew.size(dim=1)).to(self.training_device)+torch.t(chinew)@torch.inverse(G)@chinew)
            A = Ap + (torch.t(x_t1_lift) - Ap@chinew)@lambdatau@torch.t(chinew)@torch.inverse(G)  
            C = torch.t(xnew)@torch.pinverse(torch.t(x_t_lift))
          except:
            print('Please check the data, it may not be full rank or have Nah value')
          # get the K matrix in paper
          AC_mat = torch.cat((A, C), 0)
          AC_mat = np.matrix(AC_mat.cpu().detach().numpy())
          AC_mat = torch.FloatTensor(AC_mat).to(self.training_device)
          loss1, p1 = self.linearize_loss(AC_mat, x_t_lift, x_t1_lift, xnew)
          # gradient descent 
          total_loss = loss1
          if total_loss > self.eps:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
          # only saving the model with the lowest loss
          train_loss.append(total_loss.cpu().detach().numpy())
          min_loss = min(train_loss)
          if total_loss.cpu().detach().numpy() <= min_loss:   
            state = {'model_lifting': lifting.state_dict()}
            torch.save(state, (self.results_path+'nnbasis/'+str(nd)+self.model_saved_name))
            print("Saved min loss model, loss: ", total_loss.cpu().detach().numpy())
          if total_loss <= self.eps:
            state = {'model_lifting': lifting.state_dict()}
            torch.save(state, (self.results_path+'nnbasis/'+str(nd)+self.model_saved_name))
            break
      return A.cpu().detach().numpy(), C.cpu().detach().numpy()
    
    #=====================
    # loss functions
    #=====================
    def linearize_loss(self, AC_mat, x_t_lift, x_t1_lift, xnew):
      label = torch.cat((x_t1_lift.T, xnew.T), 0)
      p1 = AC_mat[0:self.dim_lifting, :] @ x_t_lift.T
      p2 = AC_mat[self.dim_lifting:, :] @ x_t_lift.T
      prediction = torch.cat((p1, p2), 0)
      lossfunc = torch.nn.MSELoss()
      loss1 = lossfunc(prediction, label)
      return loss1, p1
