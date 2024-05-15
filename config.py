'''
Description: Paper codes of 'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." 
                            Automatica 159 (2024): 111372.'

Author: Wenjian Hao, AAE, Purdue University.

This is a configuration file.

Start at: Sep 2021.
Last Revision: Jan 2022.
    
'''

import os

#---------------------------------------------- Flags ----------------------------------------------
flags = dict(
    train      = True,                         # Decide to start training or not
    keep_train =  False,   #   True            # Keep training based on the previous model
    plot_NN    = False                         # Plot the structure of the lifting network and the decoder network
)

#---------------------------------------------- Files directions ----------------------------------------------
if not os.path.exists("./SavedResults"):
    os.makedirs("./SavedResults")
files_dir = dict(
    results_path     = 'SavedResults/',
    data_name        = 'data_collecting/data/trainingdata.pkl',
    model_saved_name = 'liftnetwork.pth',
    fileastack       = 'SavedResults/Astk.pkl',
    filecstack       = 'SavedResults/Cstk.pkl',
    filehiscstack    = 'SavedResults/Hisstk.pkl', 
    xfile            = 'SavedResults/xdata.pkl',
    tvdpred          = 'SavedResults/tvdpred.pkl' # tvdmd results
)

#---------------------------------------------- Important dimensions ----------------------------------------------
dimensions = dict(
    dim_states  = 2,                # dimension of the system states
    dim_inputs  = 0,                # dimension of the control inputs
    dim_lifting = 6                 # dimension of the lifting
)

#---------------------------------------------- Training parameters ----------------------------------------------
training_parameters = dict(
    eps               = 1e-4,       # approximation accuracy
    prebatch_size     = 100,        # batch size during initialization
    pretraining_epoch = 300,        
    prelearning_rate  = 1e-3,
    predecay_rate     = 1e-4,
    training_epoch    = 160,
    learning_rate     = 1e-3,
    decay_rate        = 1e-5,
    batch_size        = 10        
)
