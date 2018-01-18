#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import tensorflow as tf
import random
import os
import sys
import copy
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
from colorama import init, Fore, Back, Style
init()
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../neural_nets/feedforward/'))
from utilities import *
from FeedForwardNeuralNetwork import *

# Seed the random variables generator:
random.seed(38)
np.random.seed(38)

def generate1stOrderTimeAffineDynSys(init_val, dt, tau, alpha):
    traj_length = int(round(tau/dt))
    
    traj = np.zeros(traj_length)
    
    x = init_val
    for t in range(traj_length):
        traj[t] = x
        x = (1 - ((dt/tau) * alpha)) * x
    
    return traj

if __name__ == "__main__":
    plt.close('all')
    base_tau = 0.5
    dt = 1/300.0
    alpha = 25.0/3.0 # similar to (1st order) canonical system's alpha
    
    plt_color_code = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    N_traj = len(plt_color_code)
    stretched_traj_length = int(round(base_tau/dt * (N_traj + 1)/2))
    
    trajs = [None] * N_traj
    data_output = [None] * N_traj
    data_input = [None] * N_traj
    
    ax_plt = [None] * 2
    plt_label = [None] * 2
    fig, (ax_plt[0], ax_plt[1]) = plt.subplots(2, sharex=True, sharey=True)
    
    for i in range(N_traj):
        tau = (i+1) * base_tau
        traj = generate1stOrderTimeAffineDynSys(1.0, dt, tau, alpha)
        trajs[i] = traj
        data_output[i] = traj[1:] - traj[:-1] # = C_{t} - C_{t-1}
        data_input[i] = ((dt/tau) * traj[:-1]) # (dt/tau) * C_{t-1}
        plt_label[0] = 'traj ' + str(i+1) + ', tau=' + str(tau) + 's'
        plt_label[1] = 'traj ' + str(i+1)
        stretched_traj = stretchTrajectory(traj, stretched_traj_length)
        ax_plt[0].plot(traj, 
                       color=plt_color_code[i],
                       label=plt_label[0])
        ax_plt[1].plot(stretched_traj, 
                       color=plt_color_code[i],
                       label=plt_label[1])
    ax_plt[0].set_title('Unstretched vs Stretched Trajectories of 1st Order Time-Affine Dynamical Systems, dt=(1/' + str(1.0/dt) + ')s')
    ax_plt[0].set_ylabel('Unstretched')
    ax_plt[1].set_ylabel('Stretched')
    ax_plt[1].set_xlabel('Time Index')
    for p in range(2):
        ax_plt[p].legend()
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
#            f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    X = np.vstack([di.reshape(di.shape[0],1) for di in data_input]).astype(np.float32)
    Y = np.vstack([do.reshape(do.shape[0],1) for do in data_output]).astype(np.float32)
    
    N_NN_reinit_trials = 1
    batch_size = 128
    TF_max_train_iters = 100001
    
    # Dropouts:
    tf_train_dropout_keep_prob = 0.5
    
    # L2 Regularization Constant
    beta = 0.0

    logs_path = "/tmp/ffnn/"
    
    NN_name = 'my_ffnn'
    
    fraction_train_dataset = 0.85
    fraction_test_dataset  = 0.075
    
    chunk_size = 1
        
    # Initial Learning Rate
    init_learning_rate = 0.001
    
    # Define Neural Network Topology
    topology = [1, 1]
    
    # Define Neural Network Activation Function
    ffnn_hidden_layer_activation_func_list = []
    
    N_data = X.shape[0]
    D_input = X.shape[1]
    D_output = Y.shape[1]
    
    print('N_data   =', N_data)
    print('D_input  =', D_input)
    print('D_output =', D_output)
    
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
    
    Y_train = Y[permuted_idx_train_dataset,:]
    Y_valid = Y[idx_valid_dataset,:]
    Y_test = Y[idx_test_dataset,:]
    
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
        tf_train_Y_batch = tf.placeholder(tf.float32, shape=[batch_size, D_output], name="tf_train_Y_batch_placeholder")
        tf_train_Y = tf.constant(Y_train, name="tf_train_Y_constant")
        tf_valid_Y = tf.constant(Y_valid, name="tf_valid_Y_constant")
        tf_test_Y = tf.constant(Y_test, name="tf_test_Y_constant")
        
        FFNN = FeedForwardNeuralNetwork(NN_name, topology, ffnn_hidden_layer_activation_func_list)
    
        # Build the Prediction Graph (that computes predictions from the inference model).
        train_batch_prediction = FFNN.performNeuralNetworkPrediction(tf_train_X_batch, tf_train_dropout_keep_prob)
    
        # Build the Training Graph (that calculate and apply gradients).
        train_op, loss = FFNN.performNeuralNetworkTraining(train_batch_prediction, tf_train_Y_batch, 
                                                           init_learning_rate, beta)
        
        # Create a summary:
        tf.summary.scalar("loss", loss)
        
        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
    
        # Predictions for the training, validation, and test data.
        train_prediction = FFNN.performNeuralNetworkPrediction(tf_train_X, 1.0)
        valid_prediction = FFNN.performNeuralNetworkPrediction(tf_valid_X, 1.0)
        test_prediction  = FFNN.performNeuralNetworkPrediction(tf_test_X, 1.0)
    
    # Run training for TF_max_train_iters and save checkpoint at the end.
    with tf.Session(graph=ff_nn_graph) as session:
        for n_NN_reinit_trial in range(N_NN_reinit_trials):
            print ("n_NN_reinit_trial = ", n_NN_reinit_trial)
            
            # Run the Op to initialize the variables.
            tf.global_variables_initializer().run()
            print("Initialized")
            
            if (FFNN.num_params < N_train_dataset):
                print("OK: FFNN.num_params=%d < %d=N_train_dataset" % (FFNN.num_params, N_train_dataset))
            else:
                print(Fore.RED + "WARNING: FFNN.num_params=%d >= %d=N_train_dataset" % (FFNN.num_params, N_train_dataset))
                print(Style.RESET_ALL)
#                sys.exit("ERROR: FFNN.num_params=%d >= %d=N_train_dataset" % (FFNN.num_params, N_train_dataset))
            
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
                batch_Y = Y_train[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_train_X_batch : batch_X, tf_train_Y_batch : batch_Y}
        
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
                    nmse_train = computeNMSE(train_prediction.eval(), Y_train)
                    nmse_valid = computeNMSE(valid_prediction.eval(), Y_valid)
                    nmse_test = computeNMSE(test_prediction.eval(), Y_test)
                    NN_model_params = FFNN.saveNeuralNetworkToMATLABMatFile()
                    print("Training             NMSE: ", nmse_train[0])
                    print("Validation           NMSE: ", nmse_valid[0])
                    print("Test                 NMSE: ", nmse_test[0])
                    print("learned             param: ", NN_model_params[NN_name+"_output_weights"][0,0])
                    print("ground truth        param: ", -alpha)
                    print("")