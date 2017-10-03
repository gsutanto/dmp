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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../neural_nets/feedforward/'))

# Seed the random variables generator:
random.seed(38)
np.random.seed(38)

from utilities import *
from FeedForwardNeuralNetwork import *

# dummy data for neural networks learning simulation/verification:
X = sio.loadmat('dummy_X.mat', struct_as_record=True)['X']
Ct_target = sio.loadmat('dummy_Ct.mat', struct_as_record=True)['Ct']

print('X.shape =', X.shape)
print('Ct_target.shape =', Ct_target.shape)

N_NN_reinit_trials = 3
batch_size = 64
N_steps  = 100001 #700001

N_data = Ct_target.shape[0]
D_input = X.shape[1]
D_output = Ct_target.shape[1]
print('N_data   =', N_data)
print('D_input  =', D_input)
print('D_output =', D_output)

# Define Neural Network Topology
# NN_topology = [D_input, D_output]                   # the simplest and resulting in the best performance!
# NN_topology = [D_input, 25, D_output]               # best with initial_learning_rate=0.1
NN_topology = [D_input, 25, 19, D_output]
# NN_topology = [D_input, 30, 20, 10, D_output]
# NN_topology = [D_input, 250, 125, 64, 32, D_output] # most complicated but NOT a good performance??? Weird, right?

# Currently turn off dropouts:
tf_train_dropout_keep_prob = 1.0

# L2 Regularization Constant
beta = 0.0

# Initial Learning Rate
init_learning_rate = 0.1

logs_path = "/tmp/ffnn/"

NN_name = 'my_ffnn'

X = X.astype(np.float32)
Ct_target = Ct_target.astype(np.float32)

fraction_train_dataset = 0.85
fraction_test_dataset  = 0.075

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
chunk_size = 7
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
Ctt_train = Ct_target[permuted_idx_train_dataset,:]
X_valid = X[idx_valid_dataset,:]
Ctt_valid = Ct_target[idx_valid_dataset,:]
X_test = X[idx_test_dataset,:]
Ctt_test = Ct_target[idx_test_dataset,:]

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
    tf_train_Ctt_batch = tf.placeholder(tf.float32, shape=[batch_size, D_output], name="tf_train_Ctt_batch_placeholder")
    tf_train_X = tf.constant(X_train, name="tf_train_X_constant")
    tf_valid_X = tf.constant(X_valid, name="tf_valid_X_constant")
    tf_test_X = tf.constant(X_test, name="tf_test_X_constant")
    
    ffnn = FeedForwardNeuralNetwork(NN_name, NN_topology)

    # Build the Prediction Graph (that computes predictions from the inference model).
    train_batch_prediction = ffnn.performNeuralNetworkPrediction(tf_train_X_batch, tf_train_dropout_keep_prob)

    # Build the Training Graph (that calculate and apply gradients).
    train_op, loss = ffnn.performNeuralNetworkTraining(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta)
    # train_op, loss, learning_rate = ffnn.performNeuralNetworkTraining(train_batch_prediction, tf_train_Ctt_batch, init_learning_rate, beta, N_steps)

    # Create a summary:
    tf.summary.scalar("loss", loss)
    # tf.summary.scalar("learning_rate", learning_rate)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    # Predictions for the training, validation, and test data.
    train_prediction = ffnn.performNeuralNetworkPrediction(tf_train_X, 1.0)
    valid_prediction = ffnn.performNeuralNetworkPrediction(tf_valid_X, 1.0)
    test_prediction  = ffnn.performNeuralNetworkPrediction(tf_test_X, 1.0)

# Test Random Vector Generation at Output Layer's Biases:
expected_rv_output_biases  = np.array([ -1.65803023e-14, -3.75096513e-15, 5.12945704e-15, -1.96647209e-16, -8.87342059e-15, 2.00303844e-14 ])

# Run training for N_steps and save checkpoint at the end.
with tf.Session(graph=ff_nn_graph) as session:
    for n_NN_reinit_trial in range(N_NN_reinit_trials):
        # Run the Op to initialize the variables.
        tf.global_variables_initializer().run()
        print("Initialized")
        
        # Testing random number generation (by the random seed), on the biases of output layer:
        N_layers = len(NN_topology)
        for i in range(N_layers-1, N_layers):
            layer_name = ffnn.getLayerName(i)
    
            with tf.variable_scope(NN_name+'_'+layer_name, reuse=True):
                weights = tf.get_variable('weights', [NN_topology[i - 1], NN_topology[i]])
                biases = tf.get_variable('biases', [NN_topology[i]])
                # print("biases.eval() = ", biases.eval())
                print("np.linalg.norm(biases - expected_rv_output_biases) = ", np.linalg.norm(biases.eval() - expected_rv_output_biases))
    
        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
        # Start the training loop.
        for step in range(N_steps):
            # Read a batch of input dataset and labels.
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (Ctt_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_X = X_train[offset:(offset + batch_size), :]
            batch_Ctt = Ctt_train[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_X_batch : batch_X, tf_train_Ctt_batch : batch_Ctt}
    
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value, tr_batch_prediction, summary = session.run([train_op, loss, train_batch_prediction, summary_op], feed_dict=feed_dict)
    
            # write log
            writer.add_summary(summary, step)
    
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, loss_value))
                print("Minibatch NMSE : ", computeNMSE(tr_batch_prediction, batch_Ctt))
                print("Training NMSE  : ", computeNMSE(train_prediction.eval(), Ctt_train))
                print("Validation NMSE: ", computeNMSE(valid_prediction.eval(), Ctt_valid))
            if (step % 5000 == 0):
                NN_model_params = ffnn.saveNeuralNetworkToMATLABMatFile()
                # print("Logging NN_model_params.mat ...")
                sio.savemat(('models/params_reinit_'+str(n_NN_reinit_trial)+'.mat'), NN_model_params)
        nmse_train = computeNMSE(train_prediction.eval(), Ctt_train)
        nmse_valid = computeNMSE(valid_prediction.eval(), Ctt_valid)
        nmse_test = computeNMSE(test_prediction.eval(), Ctt_test)
        nmse = {}
        nmse["nmse_train"] = nmse_train
        nmse["nmse_valid"] = nmse_valid
        nmse["nmse_test"] = nmse_test
        sio.savemat(('models/nmse_reinit_'+str(n_NN_reinit_trial)+'.mat'), nmse)
        print("Final Training NMSE  : ", nmse_train)
        print("Final Validation NMSE: ", nmse_valid)
        print("Final Test NMSE      : ", nmse_test)