from __future__ import print_function
import time
import numpy as np
import random
import scipy.io as sio
import tensorflow as tf
from six.moves import cPickle as pickle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../neural_nets/feedforward/pmnn/'))

# Seed the random variables generator:
random.seed(38)
np.random.seed(38)

from utilities import *
from PMNN import *

parent_path = 'models/'
reinit_selection_idx = [np.genfromtxt(parent_path+'reinit_selection_idx.txt', delimiter=' ', dtype='int').tolist()]
TF_max_train_iters = np.genfromtxt(parent_path+'TF_max_train_iters.txt', delimiter=' ', dtype='int')
savepath = '../../../../data/dmp_coupling/learn_obs_avoid/static_obs/neural_nets/pmnn/python_models/'
if not os.path.isdir(savepath):
    os.makedirs(savepath)

for prim_no in range(1, 2):
    print ("prim_no = ", prim_no)

    # dummy data for neural networks learning simulation/verification:
    X = sio.loadmat('input_data/test_unroll_prim_'+str(prim_no)+'_X_raw_obs_avoid.mat', struct_as_record=True)['X']
    Ct_target = sio.loadmat('input_data/test_unroll_prim_'+str(prim_no)+'_Ct_target_obs_avoid.mat', struct_as_record=True)['Ct_target']
    normalized_phase_kernels = sio.loadmat('input_data/test_unroll_prim_'+str(prim_no)+'_normalized_phase_PSI_mult_phase_V_obs_avoid.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V']
    
    filepath = parent_path + 'prim_' + str(prim_no) + '_params_reinit_' + str(reinit_selection_idx[prim_no-1]) + ('_step_%07d.mat' % TF_max_train_iters)
    
    print('X.shape                        =', X.shape)
    print('Ct_target.shape                =', Ct_target.shape)
    print('normalized_phase_kernels.shape =', normalized_phase_kernels.shape)
    
    N_data = Ct_target.shape[0]
    D_input = X.shape[1]
    D_output = Ct_target.shape[1]
    print('N_data   =', N_data)
    print('D_input  =', D_input)
    print('D_output =', D_output)
    
    # Define Neural Network Topology
    regular_NN_hidden_layer_topology = [100, 75]
    N_phaseLWR_kernels = normalized_phase_kernels.shape[1]
    NN_topology = [D_input] + regular_NN_hidden_layer_topology + [N_phaseLWR_kernels, D_output]
            
    regular_NN_hidden_layer_activation_func_list = ['relu', 'tanh']
    
    NN_name = 'my_PMNN_obs_avoid_fb'
    
    X = X.astype(np.float32)
    normalized_phase_kernels = normalized_phase_kernels.astype(np.float32)
    Ct_target = Ct_target.astype(np.float32)
    
    X_test = X
    nPSI_test = normalized_phase_kernels
    Ctt_test = Ct_target
    
    N_test_dataset = X_test.shape[0]
    print('N_test_dataset  =', N_test_dataset)
    
    # Build the complete graph for feeding inputs, training, and saving checkpoints.
    ff_nn_graph = tf.Graph()
    with ff_nn_graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_test_X = tf.constant(X_test, name="tf_test_X_constant")
        tf_test_nPSI = tf.constant(nPSI_test, name="tf_test_nPSI_constant")
        
        pmnn = PMNN(NN_name, D_input, 
                    regular_NN_hidden_layer_topology, regular_NN_hidden_layer_activation_func_list, 
                    N_phaseLWR_kernels, D_output, filepath, True)
        
        test_prediction  = pmnn.performNeuralNetworkPrediction(tf_test_X, tf_test_nPSI, 1.0)
    
    # Run training for N_steps and save checkpoint at the end.
    with tf.Session(graph=ff_nn_graph) as session:
        # Run the Op to initialize the variables.
        tf.global_variables_initializer().run()
        print("Initialized")
        
        Ctt_test_prediction = test_prediction.eval()
        nmse_test = computeNMSE(Ctt_test_prediction, Ctt_test)
        nmse = {}
        nmse["nmse_test"] = nmse_test
        Ctt_prediction = {}
        Ctt_prediction["Ctt_test_prediction"] = Ctt_test_prediction
        sio.savemat((savepath + '/prim_' + str(prim_no) + '_nmse_test_unroll.mat'), nmse)
        sio.savemat((savepath + '/prim_' + str(prim_no) + '_Ctt_test_prediction.mat'), Ctt_prediction)
        print("")
        print("Final Test NMSE      : ", nmse_test)
        print("")
