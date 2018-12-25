#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:00:00 2017

@author: gsutanto
"""

import scipy.io as sio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from NeuralNetwork import *

class FeedForwardNeuralNetwork(NeuralNetwork):
    'Class for feed-forward neural network.'
    
    def __init__(self, name, neural_net_topology, nn_hidden_layer_activation_func_list=[], filepath=""):
        self.name = name
        
        self.neural_net_topology = neural_net_topology
        print self.name + " Neural Network Topology:"
        print self.neural_net_topology
        
        self.N_layers = len(self.neural_net_topology)
        
        if (nn_hidden_layer_activation_func_list == []):
            self.neural_net_activation_func_list = ['identity'] * self.N_layers
        else:
            assert (len(nn_hidden_layer_activation_func_list) == (self.N_layers - 2)), "len(nn_hidden_layer_activation_func_list) must be == (self.N_layers-2)! Only activation functions of the hidden layers that need to be specified!"
            self.neural_net_activation_func_list = ['identity'] + nn_hidden_layer_activation_func_list + ['identity']
        # First Layer (Input Layer) always uses 'identity' activation function (and it does NOT matter actually; this is mainly for the sake of layer-indexing consistency...).
        assert (len(self.neural_net_activation_func_list) == self.N_layers), "len(self.neural_net_activation_func_list) must be == self.N_layers"
        print "Neural Network Activation Function List:"
        print self.neural_net_activation_func_list
        
        if (filepath == ""):
            self.num_params = self.defineNeuralNetworkModel()
        else:
            self.num_params = self.loadNeuralNetworkFromMATLABMatFile(filepath)
    
    def getLayerName(self, layer_index):
        assert layer_index > 0, "layer_index must be > 0"
        assert layer_index < self.N_layers, "layer_index must be < N_layers"
        if (layer_index == self.N_layers - 1):  # Output Layer
            layer_name = 'output'
        else:  # Hidden Layers
            layer_name = 'hidden' + str(layer_index)
        return layer_name
    
    def defineNeuralNetworkModel(self):
        """
        Define the Neural Network model.
        """
        num_params = 0
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            with tf.variable_scope(self.name+'_'+layer_name, reuse=False):
                weights = tf.get_variable('weights', [self.neural_net_topology[i-1], self.neural_net_topology[i]], initializer=tf.random_normal_initializer(0.0, 1e-14, seed=38))
                weights_dim = weights.get_shape().as_list()
                num_params += weights_dim[0] * weights_dim[1]
                # biases = tf.get_variable('biases', [self.neural_net_topology[i]], initializer=tf.constant_initializer(0.0))
                biases = tf.get_variable('biases', [self.neural_net_topology[i]], initializer=tf.random_normal_initializer(0.0, 1e-14, seed=38))
                num_params += biases.get_shape().as_list()[0]
        print("Total # of Parameters = %d" % num_params)
        return num_params
    
    def performNeuralNetworkPrediction(self, dataset, dropout_keep_prob=1.0):
        """
        Perform Neural Network Prediction on a given dataset.
        :param dataset: dataset on which prediction will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: output tensor (in output layer)
        """
        hidden_drop = dataset
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            with tf.variable_scope(self.name+'_'+layer_name, reuse=True):
                weights = tf.get_variable('weights', [self.neural_net_topology[i - 1], self.neural_net_topology[i]])
                biases = tf.get_variable('biases', [self.neural_net_topology[i]])
                
                affine_intermediate_result = tf.matmul(hidden_drop, weights) + biases
                if (self.neural_net_activation_func_list[i] == 'identity'):
                    activation_func_output = affine_intermediate_result
                elif (self.neural_net_activation_func_list[i] == 'tanh'):
                    activation_func_output = tf.nn.tanh(affine_intermediate_result)
                elif (self.neural_net_activation_func_list[i] == 'relu'):
                    activation_func_output = tf.nn.relu(affine_intermediate_result)
                else:
                    sys.exit('Unrecognized activation function: ' + self.neural_net_activation_func_list[i])
                
                if (i < self.N_layers - 1):     # Hidden Layer
                    hidden = activation_func_output
                    hidden_drop = tf.nn.dropout(hidden, dropout_keep_prob)
                else:                           # Output Layer (no Dropout here!)
                    output = activation_func_output
        return output
    
    def computeRegularizationL2Loss(self):
        # Create an operation that calculates L2 regularization loss.
        reg_l2_loss = 0
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            with tf.variable_scope(self.name+'_'+layer_name, reuse=True):
                weights = tf.get_variable('weights', [self.neural_net_topology[i - 1], self.neural_net_topology[i]])
                biases = tf.get_variable('biases', [self.neural_net_topology[i]])
                reg_l2_loss = reg_l2_loss + tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        
        return reg_l2_loss
    
    # def performNeuralNetworkTraining(self, prediction, ground_truth, initial_learning_rate, beta, N_steps):
    def performNeuralNetworkTraining(self, prediction, ground_truth, initial_learning_rate, beta):
        """
        Perform Neural Network Training.
        :param prediction: prediction made by current model on a dataset
        :param ground_truth: the ground truth of the corresponding dataset
        :param initial_learning_rate: initial learning rate
        :param beta: L2 regularization constant
        """
        # Create an operation that calculates L2 prediction loss.
        pred_l2_loss = tf.nn.l2_loss(prediction - ground_truth, name=self.name+'_'+'my_pred_L2_loss')
        reg_l2_loss = self.computeRegularizationL2Loss()
    
        loss = tf.reduce_mean(pred_l2_loss, name=self.name+'_'+'my_pred_L2_loss_mean') + (beta * reg_l2_loss)
    
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name=self.name+'_'+'global_step', trainable=False)
        # Exponentially-decaying learning rate:
        # learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, N_steps, 0.1)
        # Create the gradient descent optimizer with the given learning rate.
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # train_op = tf.train.MomentumOptimizer(learning_rate, momentum=learning_rate/4.0, use_nesterov=True).minimize(loss, global_step=global_step)
        # train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        # train_op = tf.train.AdagradOptimizer(initial_learning_rate).minimize(loss, global_step=global_step)
        # train_op = tf.train.AdadeltaOptimizer().minimize(loss, global_step=global_step)
        train_op = tf.train.RMSPropOptimizer(initial_learning_rate).minimize(loss, global_step=global_step)
        
        # return train_op, loss, learning_rate
        return train_op, loss
    
    # Save model.
    def saveNeuralNetworkToMATLABMatFile(self):
        """
        Save the Neural Network model into a MATLAB *.m file.
        """
        model_params={}
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            with tf.variable_scope(self.name+'_'+layer_name, reuse=True):
                weights = tf.get_variable('weights', [self.neural_net_topology[i - 1], self.neural_net_topology[i]])
                biases = tf.get_variable('biases', [self.neural_net_topology[i]])
                model_params[self.name+'_'+layer_name+"_weights"] = weights.eval()
                model_params[self.name+'_'+layer_name+"_biases"] = biases.eval()
    
        return model_params

    # Load model.
    def loadNeuralNetworkFromMATLABMatFile(self, filepath):
        """
        Load a Neural Network model from a MATLAB *.mat file. Functionally comparable to the defineNeuralNetworkModel() function.
        :param filepath: (relative) path in the directory structure specifying the location of the file to be loaded.
        """
        num_params = 0
        model_params = sio.loadmat(filepath, struct_as_record=True)
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
            
            with tf.variable_scope(self.name+'_'+layer_name, reuse=False):
                weights = tf.get_variable('weights', initializer=model_params[self.name+'_'+layer_name+"_weights"])
                weights_dim = weights.get_shape().as_list()
                num_params += weights_dim[0] * weights_dim[1]
                biases = tf.get_variable('biases', initializer=model_params[self.name+'_'+layer_name+"_biases"][0,:])
                num_params += biases.get_shape().as_list()[0]
        print("Total # of Parameters = %d" % num_params)
        return num_params
