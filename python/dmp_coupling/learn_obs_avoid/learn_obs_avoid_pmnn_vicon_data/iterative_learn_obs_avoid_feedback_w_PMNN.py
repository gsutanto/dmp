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
pmnn_model_parent_dir_path='../tf/models/'
pmnn_name = 'my_PMNN_obs_avoid_fb'

dmp_basis_funcs_size = 25
canonical_order = 2
ctraj_local_coordinate_frame_selection = GSUTANTO_LOCAL_COORD_FRAME
is_using_scaling = [False] * D_output # NOT using scaling on CartCoordDMP for now...
                                        
tau_sys = TauSystem(MIN_TAU)
canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
loa_parameters = TCLearnObsAvoidFeatureParameter(D_input,
                                                 dmp_basis_funcs_size, D_output,
                                                 pmnn_model_parent_dir_path, 
                                                 PMNN_MODEL, pmnn_name)
tcloa = TransformCouplingLearnObsAvoid(loa_parameters, tau_sys)
transform_couplers_list = [tcloa]
cart_coord_dmp = CartesianCoordDMP(dmp_basis_funcs_size, canonical_sys_discr, 
                                   ctraj_local_coordinate_frame_selection,
                                   transform_couplers_list)
cart_coord_dmp.setScalingUsage(is_using_scaling)
cart_coord_dmp.setParams(ccdmp_baseline_params['W'], ccdmp_baseline_params['A_learn'])

N_settings = len(data_global_coord["obs_avoid"][0])
prim_no = 0 # There is only one (1) primitive here.



N_settings_per_batch = 10
N_demos_per_setting = 1

batch_size = N_settings_per_batch * 200

unroll_dataset_Ct_obs_avoid = {}
unroll_dataset_Ct_obs_avoid["sub_X"] = [[None] * N_settings]
unroll_dataset_Ct_obs_avoid["sub_Ct_target"] = [[None] * N_settings]


# Create directories if not currently exist:
model_parent_dir_path = '../tf/models/'
reinit_selection_idx = list(np.loadtxt(model_parent_dir_path+'reinit_selection_idx.txt', dtype=np.int, ndmin=1))
TF_max_train_iters = np.loadtxt(model_parent_dir_path+'TF_max_train_iters.txt', dtype=np.int, ndmin=0)
init_model_param_filepath = model_parent_dir_path + 'prim_' + str(prim_no+1) + '_params_reinit_' + str(reinit_selection_idx[prim_no]) + ('_step_%07d.mat' % TF_max_train_iters)

regular_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1))

D_input = 17
D_output = 3
print('D_input  =', D_input)
print('D_output =', D_output)

# Define Neural Network Topology
N_phaseLWR_kernels = 25
NN_topology = [D_input] + regular_NN_hidden_layer_topology + [N_phaseLWR_kernels, D_output]


# Dropouts:
tf_train_dropout_keep_prob = 1.0

# L2 Regularization Constant
beta = 0.0

logs_path = "/tmp/pmnn/"

NN_name = 'my_PMNN_obs_avoid_fb'

is_performing_weighted_training = 1

# Initial Learning Rate
init_learning_rate = 0.001

# Phase Modulation Usage Flag
is_using_phase_kernel_modulation = True

input_X_descriptor_string = 'raw_reg_hidden_layer_100relu_75tanh'
print ("input_X_descriptor_string = ", input_X_descriptor_string)

model_output_dir_path = '../tf/models/iterative_unroll/'
if not os.path.isdir(model_output_dir_path):
    os.makedirs(model_output_dir_path)

# Build the complete graph for feeding inputs, training, and saving checkpoints.
ff_nn_graph = tf.Graph()
with ff_nn_graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_X_batch = tf.placeholder(tf.float32, shape=[batch_size, D_input], name="tf_train_X_batch_placeholder")
    tf_train_nPSI_batch = tf.placeholder(tf.float32, shape=[batch_size, N_phaseLWR_kernels], name="tf_train_nPSI_batch_placeholder")
    tf_train_W_batch = tf.placeholder(tf.float32, shape=[batch_size, 1], name="tf_train_W_batch_placeholder")
    tf_train_Ctt_batch = tf.placeholder(tf.float32, shape=[batch_size, D_output], name="tf_train_Ctt_batch_placeholder")
    
    # PMNN is initialized with parameters specified in filepath:
    pmnn = PMNN(NN_name, D_input, 
                regular_NN_hidden_layer_topology, regular_NN_hidden_layer_activation_func_list, 
                N_phaseLWR_kernels, D_output, init_model_param_filepath, is_using_phase_kernel_modulation, False)

    # Build the Prediction Graph (that computes predictions from the inference model).
    train_batch_prediction = pmnn.performNeuralNetworkPrediction(tf_train_X_batch, tf_train_nPSI_batch, tf_train_dropout_keep_prob)
    
    # Build the Training Graph (that calculate and apply gradients), per output dimension.
    if (is_performing_weighted_training):
        train_op_dim0, loss_dim0 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 0, tf_train_W_batch)
        train_op_dim1, loss_dim1 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 1, tf_train_W_batch)
        train_op_dim2, loss_dim2 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 2, tf_train_W_batch)
    else:
        train_op_dim0, loss_dim0 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 0)
        train_op_dim1, loss_dim1 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 1)
        train_op_dim2, loss_dim2 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 2)
    
    # Create a summary:
    #tf.summary.scalar("loss_dim_"+str(dim_out), loss_dim[dim_out])

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

# Test Random Vector Generation at Output Layer's Biases:
#expected_rv_output_biases  = np.array([ -1.65803023e-14, -3.75096513e-15, 5.12945704e-15, -1.96647209e-16, -8.87342059e-15, 2.00303844e-14 ])

# Run training for TF_max_train_iters and save checkpoint at the end.
with tf.Session(graph=ff_nn_graph) as session:
    # Run the Op to initialize the variables.
    tf.global_variables_initializer().run()
    print("Initialized")
    
    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Start the training loop.
    for step in range(TF_max_train_iters):
        list_batch_settings = list(np.random.permutation(N_settings))[0:N_settings_per_batch]
        
        for ns in list_batch_settings:
            N_demos = len(data_global_coord["obs_avoid"][1][ns])
            
            # the index 0 before ns seems unnecessary, but this is just for the sake of generality, if we have multiple primitives
            unroll_dataset_Ct_obs_avoid["sub_X"][prim_no][ns] = [None] * N_demos
            unroll_dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns] = [None] * N_demos
            
            for nd in range(N_demos_per_setting):
#                print ('Setting #' + str(ns+1) + ', Demo #' + str(nd+1) + '/' + str(N_demos_per_setting))
                [unroll_dataset_Ct_obs_avoid["sub_X"][prim_no][ns][nd],
                 unroll_dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns][nd],
                 _] = unrollLearnedObsAvoidViconTraj(data_global_coord["obs_avoid"][1][ns][nd],
                                                     data_global_coord["obs_avoid"][0][ns],
                                                     data_global_coord["dt"],
                                                     ccdmp_baseline_params,
                                                     cart_coord_dmp)
        
        subset_settings_indices = list_batch_settings
        subset_demos_indices = range(N_demos_per_setting)
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
                                             prim_no)
        
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
#        print('Ct_target.shape                =', Ct_target.shape)
#        print('normalized_phase_kernels.shape =', normalized_phase_kernels.shape)
#        print('data_point_priority.shape      =', data_point_priority.shape)
        
        N_data = X.shape[0]
        permuted_idx_train_dataset = list(np.random.permutation(N_data))[0:batch_size]
        
        X_train = X[permuted_idx_train_dataset,:]
        nPSI_train = normalized_phase_kernels[permuted_idx_train_dataset,:]
        Ctt_train = Ct_target[permuted_idx_train_dataset,:]
        W_train = data_point_priority[permuted_idx_train_dataset,:]
        
#        print('X_train.shape                  =', X_train.shape)

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
#         _, loss_value, tr_batch_prediction, summary = session.run([train_op, loss, train_batch_prediction, summary_op], feed_dict=feed_dict)
        _, loss_value_0, _, loss_value_1, _, loss_value_2, tr_batch_prediction = session.run([train_op_dim0, loss_dim0, train_op_dim1, loss_dim1, train_op_dim2, loss_dim2, train_batch_prediction], feed_dict=feed_dict)
        
        # write log
        #writer.add_summary(summary, step)
        
        NN_model_params = pmnn.saveNeuralNetworkToMATLABMatFile()
        tcloa.loa_param.pmnn.model_params = NN_model_params
        
        print("Step %d: NMSE = " % step, computeNMSE(tr_batch_prediction, batch_Ctt))
        if (step % 10 == 0):
            nmse = {}
            print("")
            if ((is_performing_weighted_training) and (step % 50 == 0) and (step > 0)):
                wnmse_train = computeWNMSE(tr_batch_prediction, batch_Ctt, W_train)
#                 print("Training            WNMSE: ", wnmse_train)
                nmse["wnmse_train"] = wnmse_train
            nmse_train = computeNMSE(tr_batch_prediction, batch_Ctt)
            var_ground_truth_Ctt_train = np.var(Ctt_train, axis=0)
            print("Training             NMSE: ", nmse_train)
            print("Training         Variance: ", var_ground_truth_Ctt_train)
            print("")
#             if ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32)))):
            sio.savemat((model_output_dir_path+'prim_'+str(prim_no+1)+'_params_step_%07d'%step+'.mat'), NN_model_params)
            nmse["nmse_train"] = nmse_train
            sio.savemat((model_output_dir_path+'prim_'+str(prim_no+1)+'_nmse_step_%07d'%step+'.mat'), nmse)
            var_ground_truth = {}
            var_ground_truth["var_ground_truth_Ctt_train"] = var_ground_truth_Ctt_train
            sio.savemat((model_output_dir_path+'prim_'+str(prim_no+1)+'_var_ground_truth.mat'), var_ground_truth)
    print("")
#     if (is_performing_weighted_training):
#     print("Final Training            WNMSE: ", wnmse_train)
    print("Final Training             NMSE: ", nmse_train)
    print("")