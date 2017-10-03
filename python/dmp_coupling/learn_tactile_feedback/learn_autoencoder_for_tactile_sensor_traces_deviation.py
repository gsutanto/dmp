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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../neural_nets/feedforward/autoencoder/'))

# Seed the random variables generator:
random.seed(38)
np.random.seed(38)

from utilities import *
from AutoEncoder import *

# Create directories if not currently exist:
model_output_dir_path = './autoencoder_models/'
if not os.path.isdir(model_output_dir_path):
    os.makedirs(model_output_dir_path)

N_NN_reinit_trials = 1 # 3
batch_size = 128
TF_max_train_iters = 100001

# Dropouts:
tf_train_dropout_keep_prob = 0.5

# L2 Regularization Constant
beta = 0.0

logs_path = "/tmp/autoencoder/"

NN_name = 'my_autoencoder'

fraction_train_dataset = 0.85
fraction_test_dataset  = 0.075

chunk_size = 1
    
# Initial Learning Rate
init_learning_rate = 0.001

file_check_path = 'scraping/prim_1_X_raw_scraping.mat'
assert os.path.exists(file_check_path), "file_check_path=%s does NOT exist!" % file_check_path

for prim_no in range(1, 4):
    print ("prim_no = ", prim_no)
    
    # load dataset:
    X = sio.loadmat('scraping/prim_'+str(prim_no)+'_X_raw_scraping.mat', struct_as_record=True)['X'].astype(np.float32)
    X_dim_reduced_pca = sio.loadmat('scraping/prim_'+str(prim_no)+'_X_dim_reduced_scraping.mat', struct_as_record=True)['X_dim_reduced'].astype(np.float32)
    
    print('X.shape  =', X.shape)
    
    N_data = X.shape[0]
    D_input = X.shape[1]
    
    print('N_data   =', N_data)
    print('D_input  =', D_input)
    
    # Define Neural Network Topology
    encoder_NN_hidden_layer_topology = [100]
    
    D_latent = X_dim_reduced_pca.shape[1] # match the latent dimensionality with that of PCA's result
    
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
    X_valid = X[idx_valid_dataset,:]
    X_test = X[idx_test_dataset,:]
    
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
        tf_train_X = tf.constant(X_train, name="tf_train_X_constant")
        tf_valid_X = tf.constant(X_valid, name="tf_valid_X_constant")
        tf_test_X = tf.constant(X_test, name="tf_test_X_constant")
        
        AE = AutoEncoder(NN_name, D_input, encoder_NN_hidden_layer_topology, D_latent, "")
    
        # Build the Prediction Graph (that computes predictions from the inference model).
        train_batch_prediction = AE.performNeuralNetworkPrediction(tf_train_X_batch, tf_train_dropout_keep_prob)
    
        # Build the Training Graph (that calculate and apply gradients).
        train_op, loss = AE.performNeuralNetworkTraining(train_batch_prediction, tf_train_X_batch, init_learning_rate, beta)
        
        # Create a summary:
        tf.summary.scalar("loss", loss)
        
        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
    
        # Predictions for the training, validation, and test data.
        train_prediction = AE.performNeuralNetworkPrediction(tf_train_X, 1.0)
        valid_prediction = AE.performNeuralNetworkPrediction(tf_valid_X, 1.0)
        test_prediction  = AE.performNeuralNetworkPrediction(tf_test_X, 1.0)
    
    # Run training for TF_max_train_iters and save checkpoint at the end.
    with tf.Session(graph=ff_nn_graph) as session:
        for n_NN_reinit_trial in range(N_NN_reinit_trials):
            print ("n_NN_reinit_trial = ", n_NN_reinit_trial)
            
            # Run the Op to initialize the variables.
            tf.global_variables_initializer().run()
            print("Initialized")
            
            if (AE.num_params < N_train_dataset):
                print("OK: AE.num_params=%d < %d=N_train_dataset" % (AE.num_params, N_train_dataset))
            else:
                print(Fore.RED + "WARNING: AE.num_params=%d >= %d=N_train_dataset" % (AE.num_params, N_train_dataset))
                print(Style.RESET_ALL)
#                sys.exit("ERROR: AE.num_params=%d >= %d=N_train_dataset" % (AE.num_params, N_train_dataset))
            
            # create log writer object
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=4)
            
            # Start the training loop.
            for step in range(TF_max_train_iters):
                # Read a batch of input dataset and labels.
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (X_train.shape[0] - batch_size)
                # Generate a minibatch.
                batch_X = X_train[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_train_X_batch : batch_X}
        
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value, tr_batch_prediction, summary = session.run([train_op, loss, train_batch_prediction, summary_op], feed_dict=feed_dict)
                
                # write log
                writer.add_summary(summary, step)
                
                if ((step % 5000 == 0) or ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32))))):
                    nmse = {}
                    var_ground_truth_X_train = np.var(X_train, axis=0)
                    nmse_train = computeNMSE(train_prediction.eval(), X_train)
                    nmse_valid = computeNMSE(valid_prediction.eval(), X_valid)
                    nmse_test = computeNMSE(test_prediction.eval(), X_test)
                    display_matrix = np.array([var_ground_truth_X_train,nmse_train,nmse_valid,nmse_test]);
                    display_matrix = display_matrix.transpose()
                    display_matrix = display_matrix[0:38,:]
                    display_matrix = display_matrix[list(reversed(np.argsort(display_matrix[:,0]))),:]
                    print("Minibatch loss at step %d: %f" % (step, loss_value))
                    print("Training Variance,   Training NMSE,   Validation NMSE, Test NMSE")
                    print(display_matrix)
                    print("")
#                    if ((step > 0) and ((step == np.power(10,(np.floor(np.log10(step)))).astype(np.int32)) or (step == 5 * np.power(10,(np.floor(np.log10(step/5)))).astype(np.int32)))):
                    NN_model_params = AE.saveNeuralNetworkToMATLABMatFile()
                    sio.savemat((model_output_dir_path+'prim_'+str(prim_no)+'_params_reinit_'+str(n_NN_reinit_trial)+'_step_%07d'%step+'.mat'), NN_model_params)
                    nmse["nmse_train"] = nmse_train
                    nmse["nmse_valid"] = nmse_valid
                    nmse["nmse_test"] = nmse_test
                    sio.savemat((model_output_dir_path+'prim_'+str(prim_no)+'_nmse_reinit_'+str(n_NN_reinit_trial)+'_step_%07d'%step+'.mat'), nmse)
