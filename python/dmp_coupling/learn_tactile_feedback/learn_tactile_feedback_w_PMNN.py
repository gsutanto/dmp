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
#init()
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../neural_nets/feedforward/autoencoder/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../neural_nets/feedforward/pmnn/'))

# Seed the random variables generator:
random.seed(38)
np.random.seed(38)

from utilities import *
from AutoEncoder import *
from PMNN import *

# Create directories if not currently exist:
model_parent_dir_path = './models/'
if not os.path.isdir(model_parent_dir_path):
    os.makedirs(model_parent_dir_path)

N_NN_reinit_trials = 1 # 3
batch_size = 64
TF_max_train_iters_orig = np.loadtxt(model_parent_dir_path+'TF_max_train_iters.txt', dtype=np.int, ndmin=0) + 1

# Dropouts:
tf_train_dropout_keep_prob = 0.5

# L2 Regularization Constant
beta = 0.0

logs_path = "/tmp/pmnn/"

NN_name = 'my_ffNNphaseLWR'

fraction_train_dataset = 0.85
fraction_test_dataset  = 0.075

chunk_size = np.loadtxt(model_parent_dir_path+'chunk_size.txt', dtype=np.int, ndmin=0) + 0

is_performing_weighted_training = 1
is_performing_generalization_test = 1

generalization_test_comparison_dimension = 4

#input_selector = 1 # X_raw input, Phase-Modulated Neural Network                      (PMNN) with  1 regular hidden layer only of 100 nodes and 25 nodes in the phase-modulated final hidden layer (regular execution)
#input_selector = 2 # X_dim_reduced_pca input,                                          PMNN  with NO regular hidden layer                   and 25 nodes in the phase-modulated final hidden layer (comparison: with [Chebotar & Kroemer]'s model)
#input_selector = 3 # X_raw input, Phase-Modulated Neural Network                      (PMNN) with  1 regular hidden layer only of   6 nodes and 25 nodes in the phase-modulated final hidden layer (comparison: between different number of nodes in the regular hidden layer)
#input_selector = 4 # X_dim_reduced_autoencoder input,                                  PMNN  with NO regular hidden layer                   and 25 nodes in the phase-modulated final hidden layer (comparison: with [Chebotar & Kroemer]'s model)
#input_selector = 5 # X_raw input, regular Feed-Forward Neural Network                 (FFNN) with 100 and 25 nodes in the (regular) hidden layers,           NO phase modulation                   (comparison between different neural network structures)
#input_selector = 6 # X_raw input,                                                      PMNN  with NO regular hidden layer                   and 25 nodes in the phase-modulated final hidden layer (comparison between different neural network structures)
#input_selector = 7 # X_raw_phase_X_phase_V input, regular Feed-Forward Neural Network (FFNN) with 100 and 25 nodes in the (regular) hidden layers,           NO phase modulation                   (comparison between different neural network structures)

if (is_performing_generalization_test == 1):
    input_selector_list = [1,2,3,4,5,6,7]
#    input_selector_list = [4]
else:
    input_selector_list = [1]

for input_selector in input_selector_list:
    print ("input_selector = ", input_selector)
    
    if (is_performing_generalization_test == 1):
        generalization_test_id = 1
        generalization_test_id_string = '_'+str(generalization_test_id)
        generalization_test_sub_path = 'generalization_test/'
    else:
        generalization_test_id = -1
        generalization_test_id_string = ''
        generalization_test_sub_path = ''
    
    # Initial Learning Rate
    if (input_selector == 4):
        init_learning_rate = 0.001
    else:
        init_learning_rate = 0.001
    
    if ((input_selector == 2) or (input_selector == 4)):
        TF_max_train_iters = 3 * TF_max_train_iters_orig # apparently PCA and AutoEncoder required 3 times other model's gradient descent iterations for achieving convergence ...
    else:
        TF_max_train_iters = TF_max_train_iters_orig
    
    # Phase Modulation Usage Flag
    if ((input_selector == 5) or (input_selector == 7)):
        is_using_phase_kernel_modulation = False
    else:
        is_using_phase_kernel_modulation = True
    
    if (input_selector == 1):
        input_X_descriptor_string = 'raw_reg_hidden_layer_100'
        model_output_dir_path = './models/'+generalization_test_sub_path
    elif (input_selector == 2):
        input_X_descriptor_string = 'dim_reduced_pca_no_reg_hidden_layer'
        model_output_dir_path = './models/'+generalization_test_sub_path+'comparison_vs_separated_feature_learning/input_X_'+input_X_descriptor_string+'/'
    elif (input_selector == 3):
        input_X_descriptor_string = 'raw_reg_hidden_layer_6'
        model_output_dir_path = './models/'+generalization_test_sub_path+'comparison_vs_separated_feature_learning/input_X_'+input_X_descriptor_string+'/'
    elif (input_selector == 4):
        input_X_descriptor_string = 'dim_reduced_autoencoder_no_reg_hidden_layer'
        model_output_dir_path = './models/'+generalization_test_sub_path+'comparison_vs_separated_feature_learning/input_X_'+input_X_descriptor_string+'/'
    elif (input_selector == 5):
        input_X_descriptor_string = 'raw_ffnn_hidden_layer_100_25'
        model_output_dir_path = './models/'+generalization_test_sub_path+'comparison_vs_different_neural_net_structure/input_X_'+input_X_descriptor_string+'/'
    elif (input_selector == 6):
        input_X_descriptor_string = 'raw_no_reg_hidden_layer'
        model_output_dir_path = './models/'+generalization_test_sub_path+'comparison_vs_different_neural_net_structure/input_X_'+input_X_descriptor_string+'/'
    elif (input_selector == 7):
        input_X_descriptor_string = 'raw_phase_X_phase_V_ffnn_hidden_layer_100_25'
        model_output_dir_path = './models/'+generalization_test_sub_path+'comparison_vs_different_neural_net_structure/input_X_'+input_X_descriptor_string+'/'
    print ("input_X_descriptor_string = ", input_X_descriptor_string)
    
    if not os.path.isdir(model_output_dir_path):
        os.makedirs(model_output_dir_path)
    
    file_check_path = 'scraping/'+generalization_test_sub_path+'prim_1_X_raw_scraping'+generalization_test_id_string+'.mat'
    assert os.path.exists(file_check_path), "file_check_path=%s does NOT exist!" % file_check_path
    
    while os.path.exists(file_check_path):
        if (is_performing_generalization_test == 1):
            print('Perform generalization test, id =',generalization_test_id,'...')
        else:
            print('Perform NON-generalization test training...')
        
        for prim_no in range(1, 4):
#        for prim_no in range(2, 4):
            print ("prim_no = ", prim_no)
            
            # load dataset:
            if ((input_selector == 1) or (input_selector == 3) or (input_selector == 5) or (input_selector == 6)):
                X = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_X_raw_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X'].astype(np.float32)
                X_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_X_raw_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X'].astype(np.float32)
            elif (input_selector == 7):
                X = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_X_raw_phase_X_phase_V_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X_phase_X_phase_V'].astype(np.float32)
                X_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_X_raw_phase_X_phase_V_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X_phase_X_phase_V'].astype(np.float32)
            elif (input_selector == 2):
                X = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_X_dim_reduced_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X_dim_reduced'].astype(np.float32)
                X_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_X_dim_reduced_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X_dim_reduced'].astype(np.float32)
            elif (input_selector == 4):
#                X = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_X_dim_reduced_autoencoder_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X_dim_reduced_autoencoder'].astype(np.float32)
#                X_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_X_dim_reduced_autoencoder_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X_dim_reduced_autoencoder'].astype(np.float32)
                X_raw = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_X_raw_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X'].astype(np.float32)
                X_raw_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_X_raw_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X'].astype(np.float32)
                
                X_dim_reduced_pca = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_X_dim_reduced_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['X_dim_reduced'].astype(np.float32)
                
                encoder_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
                encoder_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1))
                
                D_latent = X_dim_reduced_pca.shape[1] # match the latent dimensionality with that of PCA's result
                
                ae_name = 'my_autoencoder'
                ae_TF_max_train_iters = 100000
                ae_model_filepath = './autoencoder_models/prim_'+str(prim_no)+'_params_reinit_0_step_%07d'%ae_TF_max_train_iters+'.mat'
                
                # Build the complete graph for feeding inputs and encoding by an AutoEncoder.
                ae_graph = tf.Graph()
                with ae_graph.as_default():
                    # Input data.
                    tf_X_raw = tf.constant(X_raw, name="tf_X_raw_constant")
                    tf_X_raw_generalization_test = tf.constant(X_raw_generalization_test, name="tf_X_raw_generalization_test_constant")
                    
                    AE = AutoEncoder(ae_name, X_raw.shape[1], 
                                     encoder_NN_hidden_layer_topology, encoder_NN_hidden_layer_activation_func_list, 
                                     D_latent, ae_model_filepath)
                    
                    tf_X_latent  = AE.encode(tf_X_raw, 1.0)
                    tf_X_latent_generalization_test  = AE.encode(tf_X_raw_generalization_test, 1.0)
                
                # Run the AutoEncoder's Encoding process.
                with tf.Session(graph=ae_graph) as session:
                    # Run the Op to initialize the variables.
                    tf.global_variables_initializer().run()
                    print("Initialized AutoEncoder")
                    
                    X = tf_X_latent.eval()
                    X_generalization_test = tf_X_latent_generalization_test.eval()
            
            Ct_target = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_Ct_target_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['Ct_target'].astype(np.float32)
            normalized_phase_kernels = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_normalized_phase_PSI_mult_phase_V_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V'].astype(np.float32)
            data_point_priority = sio.loadmat('scraping/'+generalization_test_sub_path+'prim_'+str(prim_no)+'_data_point_priority_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['data_point_priority'].astype(np.float32)
            
            #Ct_target_all = Ct_target
            #Ct_target = np.zeros((Ct_target_all.shape[0],1))
            #Ct_target[:,0] = Ct_target_all[:,4]
            
            print('X.shape                        =', X.shape)
            print('Ct_target.shape                =', Ct_target.shape)
            print('normalized_phase_kernels.shape =', normalized_phase_kernels.shape)
            print('data_point_priority.shape      =', data_point_priority.shape)
            
            Ctt_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_Ct_target_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['Ct_target'].astype(np.float32)
            nPSI_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_normalized_phase_PSI_mult_phase_V_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['normalized_phase_PSI_mult_phase_V'].astype(np.float32)
            W_generalization_test = sio.loadmat('scraping/'+generalization_test_sub_path+'test_unroll_prim_'+str(prim_no)+'_data_point_priority_scraping'+generalization_test_id_string+'.mat', struct_as_record=True)['data_point_priority'].astype(np.float32)
            
            #Ctt_generalization_test_all = Ctt_generalization_test
            #Ctt_generalization_test = np.zeros((Ctt_generalization_test_all.shape[0],1))
            #Ctt_generalization_test[:,0] = Ctt_generalization_test_all[:,4]
            
            print('X_generalization_test.shape    =', X_generalization_test.shape)
            print('Ctt_generalization_test.shape  =', Ctt_generalization_test.shape)
            print('nPSI_generalization_test.shape =', nPSI_generalization_test.shape)
            print('W_generalization_test.shape    =', W_generalization_test.shape)
            
            N_data = Ct_target.shape[0]
            D_input = X.shape[1]
            D_output = Ct_target.shape[1]
            print('N_data   =', N_data)
            print('D_input  =', D_input)
            print('D_output =', D_output)
            
            # Define Neural Network Topology
            if ((input_selector == 1) or (input_selector == 5) or (input_selector == 7)):
                regular_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
            elif ((input_selector == 2) or (input_selector == 4) or (input_selector == 6)):
                regular_NN_hidden_layer_topology = []
            elif (input_selector == 3):
                regular_NN_hidden_layer_topology = [6]
            N_phaseLWR_kernels = normalized_phase_kernels.shape[1]
            NN_topology = [D_input] + regular_NN_hidden_layer_topology + [N_phaseLWR_kernels, D_output]
            
            regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1)) * len(regular_NN_hidden_layer_topology)
            
            '''
            # Basic Permutation (for Stochastic Gradient Descent)
            permutation = np.random.permutation(N_data)
            X_shuffled = X[permutation,:]
            Ct_target_shuffled = Ct_target[permutation,:]
            
            N_train_dataset = np.round(fraction_train_dataset * N_data).astype(int)
            N_test_dataset = np.round(fraction_test_dataset * N_data).astype(int)
            N_valid_dataset = N_data - N_train_dataset - N_test_dataset
            print('N_train_dataset =', N_train_dataset)
            print('N_valid_dataset =', N_valid_dataset)
            print('N_test_dataset  =', N_test_dataset)
            
            X_train = X_shuffled[0:N_train_dataset,:]
            Ctt_train = Ct_target_shuffled[0:N_train_dataset,:]
            X_valid = X_shuffled[N_train_dataset:(N_train_dataset+N_valid_dataset),:]
            Ctt_valid = Ct_target_shuffled[N_train_dataset:(N_train_dataset+N_valid_dataset),:]
            X_test = X_shuffled[(N_train_dataset+N_valid_dataset):N_data,:]
            Ctt_test = Ct_target_shuffled[(N_train_dataset+N_valid_dataset):N_data,:]
            '''
            
            # Permutation with Chunks (for Stochastic Gradient Descent (SGD))
            data_idx_chunks = list(chunks(range(N_data), chunk_size))
            N_chunks = len(data_idx_chunks)
            
            N_train_chunks = np.round(fraction_train_dataset * N_chunks).astype(int)
            N_test_chunks = np.round(fraction_test_dataset * N_chunks).astype(int)
            N_valid_chunks = N_chunks - N_train_chunks - N_test_chunks
            
            chunk_permutation = np.random.permutation(N_chunks)
            chunk_idx_train = np.sort(chunk_permutation[0:N_train_chunks], 0)
            chunk_idx_valid = np.sort(chunk_permutation[N_train_chunks:(N_train_chunks+N_valid_chunks)], 0)
            chunk_idx_test = np.sort(chunk_permutation[(N_train_chunks+N_valid_chunks):N_chunks], 0)
            idx_train_dataset = np.concatenate([data_idx_chunks[i] for i in chunk_idx_train])
            idx_valid_dataset = np.concatenate([data_idx_chunks[i] for i in chunk_idx_valid])
            idx_test_dataset = np.concatenate([data_idx_chunks[i] for i in chunk_idx_test])
            # Training Dataset Index is Permuted for Stochastic Gradient Descent (SGD)
            permuted_idx_train_dataset = idx_train_dataset[np.random.permutation(len(idx_train_dataset))]
            assert (((set(permuted_idx_train_dataset).union(set(idx_valid_dataset))).union(set(idx_test_dataset))) == set(np.arange(N_data))), "NOT all data is utilized!"
            
            X_train = X[permuted_idx_train_dataset,:]
            nPSI_train = normalized_phase_kernels[permuted_idx_train_dataset,:]
            Ctt_train = Ct_target[permuted_idx_train_dataset,:]
            W_train = data_point_priority[permuted_idx_train_dataset,:]
            
            X_valid = X[idx_valid_dataset,:]
            nPSI_valid = normalized_phase_kernels[idx_valid_dataset,:]
            Ctt_valid = Ct_target[idx_valid_dataset,:]
            W_valid = data_point_priority[idx_valid_dataset,:]
            
            X_test = X[idx_test_dataset,:]
            nPSI_test = normalized_phase_kernels[idx_test_dataset,:]
            Ctt_test = Ct_target[idx_test_dataset,:]
            W_test = data_point_priority[idx_test_dataset,:]
            
            N_train_dataset = X_train.shape[0]
            N_valid_dataset = X_valid.shape[0]
            N_test_dataset = X_test.shape[0]
            print('N_train_dataset =', N_train_dataset)
            print('N_valid_dataset =', N_valid_dataset)
            print('N_test_dataset  =', N_test_dataset)
            
            # Build the complete graph for feeding inputs, training, and saving checkpoints.
            ff_nn_graph = tf.Graph()
            with ff_nn_graph.as_default():
                # Input data. For the training data, we use a placeholder that will be fed
                # at run time with a training minibatch.
                tf_train_X_batch = tf.placeholder(tf.float32, shape=[batch_size, D_input], name="tf_train_X_batch_placeholder")
                tf_train_nPSI_batch = tf.placeholder(tf.float32, shape=[batch_size, N_phaseLWR_kernels], name="tf_train_nPSI_batch_placeholder")
                tf_train_W_batch = tf.placeholder(tf.float32, shape=[batch_size, 1], name="tf_train_W_batch_placeholder")
                tf_train_Ctt_batch = tf.placeholder(tf.float32, shape=[batch_size, D_output], name="tf_train_Ctt_batch_placeholder")
                tf_train_X = tf.constant(X_train, name="tf_train_X_constant")
                tf_train_nPSI = tf.constant(nPSI_train, name="tf_train_nPSI_constant")
                tf_train_W = tf.constant(W_train, name="tf_train_W_constant")
                tf_valid_X = tf.constant(X_valid, name="tf_valid_X_constant")
                tf_valid_nPSI = tf.constant(nPSI_valid, name="tf_valid_nPSI_constant")
                tf_valid_W = tf.constant(W_valid, name="tf_valid_W_constant")
                tf_test_X = tf.constant(X_test, name="tf_test_X_constant")
                tf_test_nPSI = tf.constant(nPSI_test, name="tf_test_nPSI_constant")
                tf_test_W = tf.constant(W_test, name="tf_test_W_constant")
                tf_generalization_test_X = tf.constant(X_generalization_test, name="tf_generalization_test_X_constant")
                tf_generalization_test_nPSI = tf.constant(nPSI_generalization_test, name="tf_generalization_test_nPSI_constant")
                tf_generalization_test_W = tf.constant(W_generalization_test, name="tf_generalization_test_W_constant")
                
                pmnn = PMNN(NN_name, D_input, 
                            regular_NN_hidden_layer_topology, regular_NN_hidden_layer_activation_func_list, 
                            N_phaseLWR_kernels, D_output, "", is_using_phase_kernel_modulation)
            
                # Build the Prediction Graph (that computes predictions from the inference model).
                train_batch_prediction = pmnn.performNeuralNetworkPrediction(tf_train_X_batch, tf_train_nPSI_batch, tf_train_dropout_keep_prob)
            
                # Build the Training Graph (that calculate and apply gradients).
            #    train_op, loss = pmnn.performNeuralNetworkTraining(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta)
                # train_op, loss, learning_rate = pmnn.performNeuralNetworkTraining(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, TF_max_train_iters)
                
                # Create a summary:
            #    tf.summary.scalar("loss", loss)
                # tf.summary.scalar("learning_rate", learning_rate)
                
                # Build the Training Graph (that calculate and apply gradients), per output dimension.
                if (is_performing_weighted_training):
                    train_op_dim0, loss_dim0 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 0, tf_train_W_batch)
                    train_op_dim1, loss_dim1 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 1, tf_train_W_batch)
                    train_op_dim2, loss_dim2 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 2, tf_train_W_batch)
                    train_op_dim3, loss_dim3 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 3, tf_train_W_batch)
                    train_op_dim4, loss_dim4 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 4, tf_train_W_batch)
                    train_op_dim5, loss_dim5 = pmnn.performNeuralNetworkWeightedTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 5, tf_train_W_batch)
                else:
                    train_op_dim0, loss_dim0 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 0)
                    train_op_dim1, loss_dim1 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 1)
                    train_op_dim2, loss_dim2 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 2)
                    train_op_dim3, loss_dim3 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 3)
                    train_op_dim4, loss_dim4 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 4)
                    train_op_dim5, loss_dim5 = pmnn.performNeuralNetworkTrainingPerDimOut(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, 5)
                
                # Create a summary:
                #tf.summary.scalar("loss_dim_"+str(dim_out), loss_dim[dim_out])
            
                # merge all summaries into a single "operation" which we can execute in a session
                summary_op = tf.summary.merge_all()
            
                # Predictions for the training, validation, and test data.
                train_prediction = pmnn.performNeuralNetworkPrediction(tf_train_X, tf_train_nPSI, 1.0)
                valid_prediction = pmnn.performNeuralNetworkPrediction(tf_valid_X, tf_valid_nPSI, 1.0)
                test_prediction  = pmnn.performNeuralNetworkPrediction(tf_test_X, tf_test_nPSI, 1.0)
                generalization_test_prediction  = pmnn.performNeuralNetworkPrediction(tf_generalization_test_X, tf_generalization_test_nPSI, 1.0)
            
            # Test Random Vector Generation at Output Layer's Biases:
            #expected_rv_output_biases  = np.array([ -1.65803023e-14, -3.75096513e-15, 5.12945704e-15, -1.96647209e-16, -8.87342059e-15, 2.00303844e-14 ])
            
            # Run training for TF_max_train_iters and save checkpoint at the end.
            with tf.Session(graph=ff_nn_graph) as session:
                for n_NN_reinit_trial in range(N_NN_reinit_trials):
                    print ("n_NN_reinit_trial = ", n_NN_reinit_trial)
                    
                    # Run the Op to initialize the variables.
                    tf.global_variables_initializer().run()
                    print("Initialized")
                    
                    if (pmnn.num_params < N_train_dataset):
                        print("OK: pmnn.num_params=%d < %d=N_train_dataset" % (pmnn.num_params, N_train_dataset))
                    else:
                        print(Fore.RED + "WARNING: pmnn.num_params=%d >= %d=N_train_dataset" % (pmnn.num_params, N_train_dataset))
                        print(Style.RESET_ALL)
        #                sys.exit("ERROR: pmnn.num_params=%d >= %d=N_train_dataset" % (pmnn.num_params, N_train_dataset))
                    
                    # Testing random number generation (by the random seed), on the biases of output layer:
                    #N_layers = len(NN_topology)
                    #for i in range(N_layers-1, N_layers):
                    #    layer_name = pmnn.getLayerName(i)
                
                    # create log writer object
                    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
                
                    # Start the training loop.
                    for step in range(TF_max_train_iters):
                        # Read a batch of input dataset and labels.
                        # Pick an offset within the training data, which has been randomized.
                        # Note: we could use better randomization across epochs.
                        offset = (step * batch_size) % (Ctt_train.shape[0] - batch_size)
                        # Generate a minibatch.
                        batch_X = X_train[offset:(offset + batch_size), :]
                        batch_nPSI = nPSI_train[offset:(offset + batch_size), :]
                        batch_W = W_train[offset:(offset + batch_size), :]
                        batch_Ctt = Ctt_train[offset:(offset + batch_size), :]
                        # Prepare a dictionary telling the session where to feed the minibatch.
                        # The key of the dictionary is the placeholder node of the graph to be fed,
                        # and the value is the numpy array to feed to it.
                        feed_dict = {tf_train_X_batch : batch_X, tf_train_nPSI_batch : batch_nPSI, tf_train_W_batch : batch_W, tf_train_Ctt_batch : batch_Ctt}
                
                        # Run one step of the model.  The return values are the activations
                        # from the `train_op` (which is discarded) and the `loss` Op.  To
                        # inspect the values of your Ops or variables, you may include them
                        # in the list passed to sess.run() and the value tensors will be
                        # returned in the tuple from the call.
            #            _, loss_value, tr_batch_prediction, summary = session.run([train_op, loss, train_batch_prediction, summary_op], feed_dict=feed_dict)
                        _, loss_value_0, _, loss_value_1, _, loss_value_2, _, loss_value_3, _, loss_value_4, _, loss_value_5, tr_batch_prediction = session.run([train_op_dim0, loss_dim0, train_op_dim1, loss_dim1, train_op_dim2, loss_dim2, train_op_dim3, loss_dim3, train_op_dim4, loss_dim4, train_op_dim5, loss_dim5, train_batch_prediction], feed_dict=feed_dict)
                        
                        # write log
                        #writer.add_summary(summary, step)
                        
                        if (is_performing_generalization_test == 0):
                            if (step % 1000 == 0):
                                print("Minibatch loss at step %d: [%f, %f, %f, %f, %f, %f]" % (step, loss_value_0, loss_value_1, loss_value_2, loss_value_3, loss_value_4, loss_value_5))
                                print("Minibatch NMSE :", computeNMSE(tr_batch_prediction, batch_Ctt))
                            if ((step % 5000 == 0) or ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32))))):
                                nmse = {}
                                print("")
                                if ((is_performing_weighted_training) and (step % 5000 == 0) and (step > 0)):
                                    wnmse_train = computeWNMSE(train_prediction.eval(), Ctt_train, W_train)
                                    wnmse_valid = computeWNMSE(valid_prediction.eval(), Ctt_valid, W_valid)
                                    wnmse_test = computeWNMSE(test_prediction.eval(), Ctt_test, W_test)
                                    wnmse_generalization_test = computeWNMSE(generalization_test_prediction.eval(), Ctt_generalization_test, W_generalization_test)
        #                            print("Training            WNMSE: ", wnmse_train)
        #                            print("Validation          WNMSE: ", wnmse_valid)
        #                            print("Test                WNMSE: ", wnmse_test)
        #                            print("Generalization Test WNMSE: ", wnmse_generalization_test)
                                    nmse["wnmse_train"] = wnmse_train
                                    nmse["wnmse_valid"] = wnmse_valid
                                    nmse["wnmse_test"] = wnmse_test
                                    nmse["wnmse_generalization_test"] = wnmse_generalization_test
                                nmse_train = computeNMSE(train_prediction.eval(), Ctt_train)
                                nmse_valid = computeNMSE(valid_prediction.eval(), Ctt_valid)
                                nmse_test = computeNMSE(test_prediction.eval(), Ctt_test)
                                nmse_generalization_test = computeNMSE(generalization_test_prediction.eval(), Ctt_generalization_test)
                                var_ground_truth_Ctt_train = np.var(Ctt_train, axis=0)
                                print("Training             NMSE: ", nmse_train)
                                print("Validation           NMSE: ", nmse_valid)
                                print("Test                 NMSE: ", nmse_test)
                                print("Generalization Test  NMSE: ", nmse_generalization_test)
                                print("Training         Variance: ", var_ground_truth_Ctt_train)
                                print("")
            #                    if ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32)))):
                                NN_model_params = pmnn.saveNeuralNetworkToMATLABMatFile()
                                sio.savemat((model_output_dir_path+'prim_'+str(prim_no)+'_params_reinit_'+str(n_NN_reinit_trial)+'_step_%07d'%step+'.mat'), NN_model_params)
                                nmse["nmse_train"] = nmse_train
                                nmse["nmse_valid"] = nmse_valid
                                nmse["nmse_test"] = nmse_test
                                nmse["nmse_generalization_test"] = nmse_generalization_test
                                sio.savemat((model_output_dir_path+'prim_'+str(prim_no)+'_nmse_reinit_'+str(n_NN_reinit_trial)+'_step_%07d'%step+'.mat'), nmse)
                                var_ground_truth = {}
                                var_ground_truth["var_ground_truth_Ctt_train"] = var_ground_truth_Ctt_train
                                sio.savemat((model_output_dir_path+'prim_'+str(prim_no)+'_var_ground_truth.mat'), var_ground_truth)
                        elif (is_performing_generalization_test == 1):
                            if ((n_NN_reinit_trial == 0) and (step == 0)):
                                best_nmse_generalization_test = computeNMSE(generalization_test_prediction.eval(), Ctt_generalization_test)
                                best_nmse_train = computeNMSE(train_prediction.eval(), Ctt_train)
                                best_nmse_valid = computeNMSE(valid_prediction.eval(), Ctt_valid)
                                best_nmse_test = computeNMSE(test_prediction.eval(), Ctt_test)
                            elif (step % 500 == 0):
                                nmse_generalization_test = computeNMSE(generalization_test_prediction.eval(), Ctt_generalization_test)
                                if (nmse_generalization_test[generalization_test_comparison_dimension] < best_nmse_generalization_test[generalization_test_comparison_dimension]):
                                    best_nmse_train = computeNMSE(train_prediction.eval(), Ctt_train)
                                    best_nmse_valid = computeNMSE(valid_prediction.eval(), Ctt_valid)
                                    best_nmse_test = computeNMSE(test_prediction.eval(), Ctt_test)
                                    best_nmse_generalization_test = nmse_generalization_test
                                print("step %d" % (step))
                                print("Best Training             NMSE: ", best_nmse_train)
                                print("Best Validation           NMSE: ", best_nmse_valid)
                                print("Best Test                 NMSE: ", best_nmse_test)
                                print("Best Generalization Test  NMSE: ", best_nmse_generalization_test)
                                print("")
                    if (is_performing_generalization_test == 0):
                        print("")
    #                    if (is_performing_weighted_training):
    #                        print("Final Training            WNMSE: ", wnmse_train)
    #                        print("Final Validation          WNMSE: ", wnmse_valid)
    #                        print("Final Test                WNMSE: ", wnmse_test)
    #                        print("Final Generalization Test WNMSE: ", wnmse_generalization_test)
                        print("Final Training             NMSE: ", nmse_train)
                        print("Final Validation           NMSE: ", nmse_valid)
                        print("Final Test                 NMSE: ", nmse_test)
                        print("Final Generalization Test  NMSE: ", nmse_generalization_test)
                        print("")
                    elif (is_performing_generalization_test == 1):
                        print("")
                        print("Final Best Training             NMSE: ", best_nmse_train)
                        print("Final Best Validation           NMSE: ", best_nmse_valid)
                        print("Final Best Test                 NMSE: ", best_nmse_test)
                        print("Final Best Generalization Test  NMSE: ", best_nmse_generalization_test)
                        print("")
                        best_nmse = {}
                        best_nmse["best_nmse_train"] = best_nmse_train
                        best_nmse["best_nmse_valid"] = best_nmse_valid
                        best_nmse["best_nmse_test"] = best_nmse_test
                        best_nmse["best_nmse_generalization_test"] = best_nmse_generalization_test
                        sio.savemat((model_output_dir_path+'prim_'+str(prim_no)+'_best_nmse_trial_'+str(generalization_test_id)+'.mat'), best_nmse)
        if (is_performing_generalization_test == 1):
            generalization_test_id = generalization_test_id + 1
            generalization_test_id_string = '_'+str(generalization_test_id)
            file_check_path = 'scraping/'+generalization_test_sub_path+'prim_1_X_raw_scraping'+generalization_test_id_string+'.mat'
        else:
            break
