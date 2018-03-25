#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:00:00 2018

@author: gsutanto
@remarks: a variant of Iterative-Learning-and-Unrolling
          as implemented in iterative_learn_unroll_obs_avoid_feedback_w_PMNN.py,
          but learning-and-unrolling is done iteratively on
          the 1st 5%, 10%, 15%, ..., 100% (increasing percentage over batch)
          of each trajectory segment.
"""

from __future__ import print_function
import time
import numpy as np
import random
import scipy.io as sio
import tensorflow as tf
from six.moves import cPickle as pickle
import os
import sys
from colorama import init, Fore, Back, Style
init()
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/learn_obs_avoid/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../neural_nets/feedforward/pmnn/'))

# Seed the random variables generator:
random.seed(38)
np.random.seed(38)

from DMPTrajectory import *
from DMPState import *
from TauSystem import *
from DMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from CartesianCoordTransformer import *
from CartesianCoordDMP import *
from TCLearnObsAvoidFeatureParameter import *
from TransformCouplingLearnObsAvoid import *
from convertDemoToSupervisedObsAvoidFbDataset import *
from unrollLearnedObsAvoidViconTraj import *
from DataStacking import *
from utilities import *
from PMNN import *



frac_max_ave_batch_nmse = 0.50
final_max_ave_batch_nmse = 0.25

is_performing_iterative_traj_fraction_inclusion = True
is_continuing_from_a_specific_iter = False
is_using_offline_pretrained_model = True

if (is_performing_iterative_traj_fraction_inclusion):
    if (is_continuing_from_a_specific_iter):
        # user-specified values here:
        init_step = 2196
        init_n_fraction_data_pts_included_per_demo = 15
    else:
        init_step = 0
        init_n_fraction_data_pts_included_per_demo = 0
    N_fraction_data_pts_included_per_demo = 20
else:
    if (is_continuing_from_a_specific_iter):
        # user-specified values here:
        init_step = 2196
    else:
        init_step = 0
    init_n_fraction_data_pts_included_per_demo = 0
    N_fraction_data_pts_included_per_demo = 1

# Dropouts:
tf_train_dropout_keep_prob = 1.0

# L2 Regularization Constant
beta = 0.0

logs_path = "/tmp/pmnn/iter_learn_unroll/frac/"

is_performing_weighted_training = 1

# Initial Learning Rate
init_learning_rate = 0.001

# Phase Modulation Usage Flag
is_using_phase_kernel_modulation = True


## Demo Dataset Loading
data_global_coord = loadObj('data_multi_demo_vicon_static_global_coord.pkl')
# end of Demo Dataset Loading

## Baseline Primitive Loading
dmp_baseline_params = loadObj('dmp_baseline_params_obs_avoid.pkl')
# end of Baseline Primitive Loading

ccdmp_baseline_params = dmp_baseline_params["cart_coord"][0]

## Supervised Obstacle Avoidance Feedback Dataset Loading
dataset_Ct_obs_avoid = loadObj('dataset_Ct_obs_avoid.pkl')
# end of Supervised Obstacle Avoidance Feedback Dataset Loading

D_input = 17
D_output = 3
print('D_input  =', D_input)
print('D_output =', D_output)
pmnn_model_parent_dir_path='../tf/models/'
pmnn_model_file_path = None
pmnn_name = 'my_PMNN_obs_avoid_fb'

dmp_basis_funcs_size = 25
canonical_order = 2
ctraj_local_coordinate_frame_selection = GSUTANTO_LOCAL_COORD_FRAME
is_using_scaling = [False] * D_output # NOT using scaling on CartCoordDMP for now...
                                        
tau_sys = TauSystem(data_global_coord["dt"], MIN_TAU)
canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
loa_parameters = TCLearnObsAvoidFeatureParameter(D_input,
                                                 dmp_basis_funcs_size, D_output,
                                                 pmnn_model_parent_dir_path, 
                                                 pmnn_model_file_path,
                                                 PMNN_MODEL, pmnn_name)
tcloa = TransformCouplingLearnObsAvoid(loa_parameters, tau_sys)
transform_couplers_list = [tcloa]
cart_coord_dmp = CartesianCoordDMP(dmp_basis_funcs_size, canonical_sys_discr, 
                                   ctraj_local_coordinate_frame_selection,
                                   transform_couplers_list)
cart_coord_dmp.setScalingUsage(is_using_scaling)
cart_coord_dmp.setParams(ccdmp_baseline_params['W'], ccdmp_baseline_params['A_learn'])

model_parent_dir_path = '../tf/models/'

selected_settings_indices_file_path = model_parent_dir_path + 'selected_settings_indices.txt'
if not os.path.isfile(selected_settings_indices_file_path):
    N_settings = len(data_global_coord["obs_avoid"][0])
    selected_settings_indices = range(N_settings)
else:
    selected_settings_indices = [(i-1) for i in list(np.loadtxt(selected_settings_indices_file_path, dtype=np.int, ndmin=1))] # file is saved following MATLAB's convention (1~222)
    N_settings = len(selected_settings_indices)

print('N_settings = ' + str(N_settings))
prim_no = 0 # There is only one (1) primitive here.



min_N_settings_per_batch = 10 # minimum number of included settings per batch
N_demos_per_setting = 1
max_N_data_pts_included_per_demo = 200 # maximum number of data points included per demonstrated trajectory

batch_size = min_N_settings_per_batch * N_demos_per_setting * max_N_data_pts_included_per_demo

N_all_settings = len(data_global_coord["obs_avoid"][0])
unroll_dataset_Ct_obs_avoid = {}
unroll_dataset_Ct_obs_avoid["sub_X"] = [[None] * N_all_settings]
unroll_dataset_Ct_obs_avoid["sub_Ct_target"] = [[None] * N_all_settings]


# Create directories if not currently exist:
reinit_selection_idx = list(np.loadtxt(model_parent_dir_path+'reinit_selection_idx.txt', dtype=np.int, ndmin=1))
TF_max_train_iters = np.loadtxt(model_parent_dir_path+'TF_max_train_iters.txt', dtype=np.int, ndmin=0)
if (is_continuing_from_a_specific_iter):
    init_model_param_filepath = model_parent_dir_path + 'iterative_learn_unroll/' + 'prim_' + str(prim_no+1) + ('_params_step_%07d.mat' % init_step)
else:
    if (is_using_offline_pretrained_model):
        init_model_param_filepath = model_parent_dir_path + 'prim_' + str(prim_no+1) + '_params_reinit_' + str(reinit_selection_idx[prim_no]) + ('_step_%07d.mat' % TF_max_train_iters)
    else:
        init_model_param_filepath = ""

print ('')
print ('reinit_selection_idx      = ', reinit_selection_idx)
print ('TF_max_train_iters        = ', TF_max_train_iters)
print ('init_model_param_filepath = ', init_model_param_filepath)
print ('')

regular_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1))

# Define Neural Network Topology
N_phaseLWR_kernels = dmp_basis_funcs_size
NN_topology = [D_input] + regular_NN_hidden_layer_topology + [N_phaseLWR_kernels, D_output]


input_X_descriptor_string = 'raw_reg_hidden_layer_100relu_75tanh'
print ("input_X_descriptor_string = ", input_X_descriptor_string)

model_output_dir_path = '../tf/models/iterative_learn_unroll/'
if not os.path.isdir(model_output_dir_path):
    os.makedirs(model_output_dir_path)



# Build the complete graph for feeding inputs, training, and saving checkpoints.
pmnn_graph = tf.Graph()
with pmnn_graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_X_batch = tf.placeholder(tf.float32, shape=[batch_size, D_input], name="tf_train_X_batch_placeholder")
    tf_train_nPSI_batch = tf.placeholder(tf.float32, shape=[batch_size, N_phaseLWR_kernels], name="tf_train_nPSI_batch_placeholder")
    tf_train_W_batch = tf.placeholder(tf.float32, shape=[batch_size, 1], name="tf_train_W_batch_placeholder")
    tf_train_Ctt_batch = tf.placeholder(tf.float32, shape=[batch_size, D_output], name="tf_train_Ctt_batch_placeholder")
    
    # PMNN is initialized with parameters specified in filepath:
    pmnn = PMNN(pmnn_name, D_input, 
                regular_NN_hidden_layer_topology, regular_NN_hidden_layer_activation_func_list, 
                N_phaseLWR_kernels, D_output, init_model_param_filepath, is_using_phase_kernel_modulation, False)

    # Build the Prediction Graph (that computes predictions from the inference model).
    train_batch_prediction = pmnn.performNeuralNetworkPrediction(tf_train_X_batch, tf_train_nPSI_batch, tf_train_dropout_keep_prob)
    
    # Build the Training Graph (that calculate and apply gradients), per output dimension.
    if (is_performing_weighted_training):
        [train_op_dim0, loss_dim0] = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 0, tf_train_W_batch)
        [train_op_dim1, loss_dim1] = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 1, tf_train_W_batch)
        [train_op_dim2, loss_dim2] = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 2, tf_train_W_batch)
    else:
        [train_op_dim0, loss_dim0] = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 0)
        [train_op_dim1, loss_dim1] = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 1)
        [train_op_dim2, loss_dim2] = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 2)
    
    # Create a summary:
    tf.summary.scalar("loss_dim0", loss_dim0)
    tf.summary.scalar("loss_dim1", loss_dim1)
    tf.summary.scalar("loss_dim2", loss_dim2)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

# Run training for TF_max_train_iters and save checkpoint at the end.
with tf.Session(graph=pmnn_graph) as session:
    # Run the Op to initialize the variables.
    tf.global_variables_initializer().run()
    print("Initialized")
    
    # create log writer object
    recreateDir(logs_path)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
    batch_var_ground_truth_log = np.zeros((TF_max_train_iters, D_output))
    batch_nmse_train_log = np.zeros((TF_max_train_iters, D_output))
    average_batch_nmse_train_log = np.zeros((TF_max_train_iters, D_output))
    if (is_performing_weighted_training):
        batch_wnmse_train_log = np.zeros((TF_max_train_iters, D_output))
        average_batch_wnmse_train_log = np.zeros((TF_max_train_iters, D_output))
    fraction_increment_log = np.zeros((N_fraction_data_pts_included_per_demo, 3)) # 1st column is the fraction, 2nd column is the starting step of this fraction, 3rd column is how many learning steps it take on this fraction until reaching convergence.
    
    if (is_continuing_from_a_specific_iter):
        batch_var_ground_truth_log[0:init_step,:] = np.loadtxt(model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_var_ground_truth_log.txt')[0:init_step,:]
        batch_nmse_train_log[0:init_step,:] = np.loadtxt(model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_nmse_train_log.txt')[0:init_step,:]
        average_batch_nmse_train_log[0:init_step,:] = np.loadtxt(model_output_dir_path+'prim_'+str(prim_no+1)+'_average_batch_nmse_train_log.txt')[0:init_step,:]
        if (is_performing_weighted_training):
            batch_wnmse_train_log[0:init_step,:] = np.loadtxt(model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_wnmse_train_log.txt')[0:init_step,:]
            average_batch_wnmse_train_log[0:init_step,:] = np.loadtxt(model_output_dir_path+'prim_'+str(prim_no+1)+'_average_batch_wnmse_train_log.txt')[0:init_step,:]
        fraction_increment_log = np.loadtxt(model_output_dir_path+'prim_'+str(prim_no+1)+'_fraction_increment_log.txt')
    
    n_fraction_data_pts_included_per_demo = init_n_fraction_data_pts_included_per_demo
    
    acceptable_ave_batch_nmse = final_max_ave_batch_nmse

    # Start the training loop.
    for step in range(init_step, TF_max_train_iters):
        if ((step == init_step) or (np.max(average_batch_nmse_train_log[step-1, :]) < acceptable_ave_batch_nmse)):
            start_step_frac = step
            
            if (n_fraction_data_pts_included_per_demo < N_fraction_data_pts_included_per_demo):
                n_fraction_data_pts_included_per_demo = n_fraction_data_pts_included_per_demo + 1
            assert ((n_fraction_data_pts_included_per_demo > 0) and (n_fraction_data_pts_included_per_demo <= N_fraction_data_pts_included_per_demo))
            fraction_data_pts_included_per_demo = (1.0 * n_fraction_data_pts_included_per_demo) / N_fraction_data_pts_included_per_demo
            
            if (n_fraction_data_pts_included_per_demo < N_fraction_data_pts_included_per_demo):
                acceptable_ave_batch_nmse = frac_max_ave_batch_nmse
            else:
                acceptable_ave_batch_nmse = final_max_ave_batch_nmse
        
            N_settings_per_batch = int(round(min_N_settings_per_batch / fraction_data_pts_included_per_demo))
            # In average, each demonstrated trajectory will contribute this many data points:
            # N_data_pts_included_per_demo = int(round(max_N_data_pts_included_per_demo * fraction_data_pts_included_per_demo))
            nmse_averaging_window = 2 * int(np.ceil(N_all_settings * 1.0 / N_settings_per_batch))
            
            fraction_increment_log[(n_fraction_data_pts_included_per_demo-1),0] = fraction_data_pts_included_per_demo
            fraction_increment_log[(n_fraction_data_pts_included_per_demo-1),1] = start_step_frac
            if ((step > 0) and (fraction_increment_log.shape[0] > 1)):
                fraction_increment_log[(n_fraction_data_pts_included_per_demo-2),2] = step - fraction_increment_log[(n_fraction_data_pts_included_per_demo-2),1]
                assert (fraction_increment_log[(n_fraction_data_pts_included_per_demo-2),2] > 0.0)
        
        actual_nmse_averaging_window = min(((step - start_step_frac) + 1), nmse_averaging_window)
        assert (actual_nmse_averaging_window > 0)
        
        list_batch_settings = [ selected_settings_indices[i] for i in list(np.random.permutation(N_settings))[0:N_settings_per_batch] ]
        list_batch_setting_demos = list(np.random.permutation(3))[0:N_demos_per_setting]
        
        for ns in list_batch_settings:
            N_demos = len(data_global_coord["obs_avoid"][1][ns])
            
            # the index 0 before ns seems unnecessary, but this is just for the sake of generality, if we have multiple primitives
            unroll_dataset_Ct_obs_avoid["sub_X"][prim_no][ns] = [None] * N_demos
            unroll_dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns] = [None] * N_demos
            
            for nd in list_batch_setting_demos:
#                print ('Setting #' + str(ns+1) + ', Demo #' + str(nd+1) + '/' + str(N_demos_per_setting))
                [unroll_dataset_Ct_obs_avoid["sub_X"][prim_no][ns][nd],
                 unroll_dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns][nd],
                 _] = unrollLearnedObsAvoidViconTraj(data_global_coord["obs_avoid"][1][ns][nd],
                                                     data_global_coord["obs_avoid"][0][ns],
                                                     data_global_coord["dt"],
                                                     ccdmp_baseline_params,
                                                     cart_coord_dmp,
                                                     True,
                                                     fraction_data_pts_included_per_demo)
        
        subset_settings_indices = list_batch_settings
        subset_demos_indices = list_batch_setting_demos
        mode_stack_dataset = 2
        feature_type = 'raw'
        
        [_,
         Ct_target,
         normalized_phase_kernels, 
         data_point_priority] = stackDataset(dataset_Ct_obs_avoid, 
                                             subset_settings_indices, 
                                             mode_stack_dataset, 
                                             subset_demos_indices, 
                                             feature_type, 
                                             prim_no,
                                             fraction_data_pts_included_per_demo)
        
        [X,
         Ct_unroll,
         _,
         _] = stackDataset(unroll_dataset_Ct_obs_avoid, 
                           subset_settings_indices, 
                           mode_stack_dataset, 
                           subset_demos_indices, 
                           feature_type, 
                           prim_no)
        nmse_unroll = computeNMSE(Ct_unroll, Ct_target)
#        print ('nmse_unroll        = ' + str(nmse_unroll))
        
#        print('X.shape                        =', X.shape)
#        print('Ct_unroll.shape                =', Ct_unroll.shape)
#        print('Ct_target.shape                =', Ct_target.shape)
#        print('normalized_phase_kernels.shape =', normalized_phase_kernels.shape)
#        print('data_point_priority.shape      =', data_point_priority.shape)
        
        N_data = X.shape[0]
        permuted_idx_train_dataset = list(np.random.permutation(N_data))[0:batch_size]
        
        X_train = X[permuted_idx_train_dataset,:]
        nPSI_train = normalized_phase_kernels[permuted_idx_train_dataset,:]
        Ctt_train = Ct_target[permuted_idx_train_dataset,:]
        W_train = data_point_priority[permuted_idx_train_dataset,:]

        batch_X = X_train
        batch_nPSI = nPSI_train
        batch_W = W_train
        batch_Ctt = Ctt_train
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_X_batch : batch_X, tf_train_nPSI_batch : batch_nPSI, tf_train_W_batch : batch_W, tf_train_Ctt_batch : batch_Ctt}

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        
        [tr_batch_prediction,
         _, _, 
         _, _, 
         _, _, 
         summary] = session.run([train_batch_prediction, 
                                 train_op_dim0, loss_dim0, 
                                 train_op_dim1, loss_dim1, 
                                 train_op_dim2, loss_dim2, 
                                 summary_op], feed_dict=feed_dict)
        
        # write log
        writer.add_summary(summary, step)
        
        NN_model_params = pmnn.saveNeuralNetworkToMATLABMatFile()
        tcloa.loa_param.pmnn.model_params = NN_model_params
        
        batch_var_ground_truth_log[step, :] = np.var(Ctt_train, axis=0)
        batch_nmse_train_log[step, :] = computeNMSE(tr_batch_prediction, batch_Ctt)
        average_batch_nmse_train_log[step, :] = np.mean(batch_nmse_train_log[(step-(actual_nmse_averaging_window-1)):(step+1),:], axis=0)
        if (is_performing_weighted_training):
            batch_wnmse_train_log[step, :] = computeWNMSE(tr_batch_prediction, batch_Ctt, W_train)
            average_batch_wnmse_train_log[step, :] = np.mean(batch_wnmse_train_log[(step-(actual_nmse_averaging_window-1)):(step+1),:], axis=0)
        if ((step > 0) and (step % nmse_averaging_window == 0)):
            sio.savemat((model_output_dir_path+'prim_'+str(prim_no+1)+'_params_step_%07d'%step+'.mat'), NN_model_params)
            print ("Fraction %f, Step %d: Average " % (fraction_data_pts_included_per_demo, step) +str(actual_nmse_averaging_window)+"-last Batch Training NMSE = ", average_batch_nmse_train_log[step, :])
        if (step % 50 == 0):
            np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_var_ground_truth_log.txt'), batch_var_ground_truth_log[0:(step+1),:])
            np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_nmse_train_log.txt'), batch_nmse_train_log[0:(step+1),:])
            np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_average_batch_nmse_train_log.txt'), average_batch_nmse_train_log[0:(step+1),:])
            if (is_performing_weighted_training):
                np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_wnmse_train_log.txt'), batch_wnmse_train_log[0:(step+1),:])
                np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_average_batch_wnmse_train_log.txt'), average_batch_wnmse_train_log[0:(step+1),:])
            np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_fraction_increment_log.txt'), fraction_increment_log)
        if ((n_fraction_data_pts_included_per_demo == N_fraction_data_pts_included_per_demo) and (np.max(average_batch_nmse_train_log[step, :]) < acceptable_ave_batch_nmse)):
            break
    
    fraction_increment_log[(n_fraction_data_pts_included_per_demo-1),2] = step - fraction_increment_log[(n_fraction_data_pts_included_per_demo-1),1]
    assert (fraction_increment_log[(n_fraction_data_pts_included_per_demo-1),2] > 0)
    sio.savemat((model_output_dir_path+'prim_'+str(prim_no+1)+'_params_step_%07d'%step+'.mat'), NN_model_params)
    np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_var_ground_truth_log.txt'), batch_var_ground_truth_log[0:(step+1),:])
    np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_nmse_train_log.txt'), batch_nmse_train_log[0:(step+1),:])
    np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_average_batch_nmse_train_log.txt'), average_batch_nmse_train_log[0:(step+1),:])
    if (is_performing_weighted_training):
        np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_batch_wnmse_train_log.txt'), batch_wnmse_train_log[0:(step+1),:])
        np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_average_batch_wnmse_train_log.txt'), average_batch_wnmse_train_log[0:(step+1),:])
    np.savetxt((model_output_dir_path+'prim_'+str(prim_no+1)+'_fraction_increment_log.txt'), fraction_increment_log)
    print("")
    print("Final Average Batch Training NMSE at Step %d: " % step, batch_nmse_train_log[step, :])
    if (is_performing_weighted_training):
        print("Final Average Batch Training WNMSE at Step %d: " % step, batch_wnmse_train_log[step, :])
    print("")