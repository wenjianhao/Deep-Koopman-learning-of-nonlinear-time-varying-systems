'''
Description: Paper codes of 'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." 
                            Automatica 159 (2024): 111372.'

Author: Wenjian Hao, AAE, Purdue University.

This is a file containing all the functions.

Start at: Sep 2021.

Last Revision: Jan 2022.
'''

import torch

import numpy as np
import torch.optim as optim

from LNN import LNN
from torchsummary import summary

class DKTV_training(object):
    '''
    Deep Koopman learning for nonlinear time-varying systems
    '''
    def __init__(self, config: dict):
      # load the configureation parameters
      self.plot_NN          = config.flags['plot_NN']
      self.keep_train       = config.flags['keep_train']
      self.results_path     = config.files_dir['results_path']
      self.data_name        = config.files_dir['data_name']
      self.model_saved_name = config.files_dir['model_saved_name'] 
      self.dim_states       = config.dimensions['dim_states']
      self.dim_lifting      = config.dimensions['dim_lifting']
      self.eps              = config.training_parameters['eps']
      self.decay_rate       = config.training_parameters['decay_rate']
      self.learning_rate    = config.training_parameters['learning_rate']
      self.training_epoch   = config.training_parameters['training_epoch']
      self.training_device  = torch.device("cpu")
      # training devices, uncomment it to use gpu
      # print("Torch version: {}".format(torch.__version__))
      # cuda = torch.cuda.is_available()
      # self.training_device = torch.device("cuda" if cuda else "cpu")
      # print("Training device {}".format(self.training_device))
      
    def pretrain_model(self, X_mat, bar_X_mat):
      # some flags
      # keep training?
      if self.keep_train:
        load_liftNN = self.results_path + self.model_saved_name
        ckpt_liftNN = torch.load(load_liftNN, map_location=self.training_device)
        lifting     = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
        lifting.load_state_dict(ckpt_liftNN['model_lifting'])
        print('Keep Training')
      else:
        lifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
      # plot DNNs structure
      if self.plot_NN:
          device       = self.training_device
          modellifting = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting).to(device)
          summary(modellifting, (1,self.dim_states))
      # get the training data and pass it to the training GPU
      X_mat     = torch.t(torch.FloatTensor(X_mat).to(self.training_device))
      bar_X_mat = torch.t(torch.FloatTensor(bar_X_mat).to(self.training_device))
      lifting.to(self.training_device)
      # choose the training optimizer
      optimizer = optim.Adam(lifting.parameters(), lr=self.learning_rate, weight_decay = self.decay_rate)
      
      #---------------------------------------------- Training loop ----------------------------------------------
      train_loss = []
      lifting.train()
      for i in range(self.training_epoch):
          # lifting xt and x(t+1)
          G_mat     = lifting(X_mat)
          bar_G_mat = lifting(bar_X_mat)
          # get the A, C matrices
          A_mat = torch.t(bar_G_mat)@torch.pinverse(torch.t(G_mat))
          C_mat = X_mat.T@torch.pinverse(G_mat.T)
          # get the K matrix in paper
          AC_mat = torch.cat((A_mat, C_mat), 0).to(self.training_device)
          # loss1 is the linear lifting loss
          L_f        = self.linearize_loss(AC_mat.detach(), G_mat, bar_G_mat, X_mat)
          total_loss = L_f
          # gradient descent
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()
          # only saving the model with the lowest loss
          train_loss.append(total_loss.cpu().detach().numpy()) 
          if total_loss.cpu().detach().numpy() <= min(train_loss):
              data_his_c = torch.inverse(G_mat.T@G_mat)    
              state      = {'model_lifting': lifting.state_dict()}
              torch.save(state, (self.results_path+'nnbasis/'+str(0)+self.model_saved_name))
              print("Saved min loss model, loss: ", total_loss.cpu().detach().numpy())
          # break if the esitimation accuracy is satisfied
          if total_loss <= self.eps:
              state = {'model_lifting': lifting.state_dict()}
              torch.save(state, (self.results_path+'nnbasis/'+str(0)+self.model_saved_name))
              break
      return A_mat.cpu().detach().numpy(), C_mat.cpu().detach().numpy(), data_his_c.cpu().detach().numpy()

    def DKTV(self, Astk, Cstk, Hisstk, X_mat, bar_X_mat, batchsize, nd):
      # load pretrained DNN
      load_liftNN = self.results_path +'nnbasis/'+str(nd-1) + self.model_saved_name
      ckpt_liftNN = torch.load(load_liftNN, map_location=self.training_device)
      lifting     = LNN(dim_input=self.dim_states, dim_output=self.dim_lifting)
      lifting.load_state_dict(ckpt_liftNN['model_lifting'])
      print('Finish loading the pretrained model')
      # pass data to the training device
      X_mat     = torch.t(torch.FloatTensor(X_mat).to(self.training_device))
      bar_X_mat = torch.t(torch.FloatTensor(bar_X_mat).to(self.training_device))
      lifting.to(self.training_device)
      # split the data by time
      xnew = X_mat[(X_mat.size(dim=0)-batchsize-1):, :]
      ynew = bar_X_mat[(X_mat.size(dim=0)-batchsize-1):, :]
      # previous solution
      A_mat_tau   = torch.FloatTensor(Astk[:,:,(nd-1)]).to(self.training_device)
      C_mat_tau   = torch.FloatTensor(Cstk[:,:,(nd-1)]).to(self.training_device)
      His_mat_tau = torch.FloatTensor(Hisstk[:,:,(nd-1)]).to(self.training_device)
      # choose the training optimizer
      optimizer  = optim.Adam(lifting.parameters(), lr=self.learning_rate, weight_decay = self.decay_rate)
      train_loss = []
      lifting.train()
      # lifting xt and x(t+1) and update the dynamics matrices
      G_mat      = lifting(xnew)
      bar_G_mat  = lifting(ynew)          
      lambdatau  = torch.inverse(torch.eye(G_mat.T.size(dim=1)).to(self.training_device)+G_mat@His_mat_tau@G_mat.T)
      A_mat_tau1 = A_mat_tau + (torch.t(bar_G_mat) - A_mat_tau@G_mat.T)@lambdatau@torch.t(G_mat.T)@His_mat_tau
      C_mat_tau1 = C_mat_tau + (torch.t(xnew) - C_mat_tau@G_mat.T)@lambdatau@G_mat@His_mat_tau
      # get the K matrix in paper
      AC_mat = torch.cat((A_mat_tau1, C_mat_tau1), 0).to(self.training_device)
      
      #---------------------------------------------- Training loop ----------------------------------------------
      for i in range(self.training_epoch):
          # lifting xt and x(t+1)
          G_mat     = lifting(xnew)
          bar_G_mat = lifting(ynew)          
          L_f       = self.linearize_loss(AC_mat.detach(), G_mat, bar_G_mat, xnew)
          # gradient descent, add other loss functions if you have
          total_loss = L_f
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()
          # only saving the model with the lowest loss
          train_loss.append(total_loss.cpu().detach().numpy())
          if total_loss.cpu().detach().numpy() <= min(train_loss): 
            data_his_c = torch.inverse(G_mat.T@G_mat)  
            state      = {'model_lifting': lifting.state_dict()}
            torch.save(state, (self.results_path+'nnbasis/'+str(nd)+self.model_saved_name))
            print("Saved min loss model, loss: ", total_loss.cpu().detach().numpy())
          if total_loss <= self.eps:
            data_his_c = torch.inverse(G_mat.T@G_mat)
            state      = {'model_lifting': lifting.state_dict()}
            torch.save(state, (self.results_path+'nnbasis/'+str(nd)+self.model_saved_name))
            break
      return A_mat_tau1.cpu().detach().numpy(), C_mat_tau1.cpu().detach().numpy(), data_his_c.cpu().detach().numpy()

    #---------------------------------------------- Loss functions ----------------------------------------------
    def linearize_loss(self, AC_mat, G_mat, bar_G_mat, xnew):
      label      = torch.cat((bar_G_mat.T, xnew.T), 0)
      p1         = AC_mat[0:self.dim_lifting, :] @ G_mat.T
      p2         = AC_mat[self.dim_lifting:, :] @ G_mat.T
      prediction = torch.cat((p1, p2), 0)
      lossfunc   = torch.nn.MSELoss()
      L_f        = lossfunc(prediction, label)
      return L_f
