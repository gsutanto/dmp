#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:00:00 2017

@author: gsutanto
@comment: version 2: Using tf.layers library and Batch Normalization
"""

import scipy.io as sio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from NeuralNetwork import *

class FeedForwardNeuralNetworkV2(NeuralNetwork):
    'Class for feed-forward neural network (version 2).'
    
    def __init__(self, name, neural_net_topology, nn_hidden_layer_activation_func_list=[], is_using_batch_normalization=True):
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
        
        self.num_params = self.countNeuralNetworkModelNumParams()
        
        self.is_using_batch_normalization = is_using_batch_normalization
    
    def countNeuralNetworkModelNumParams(self):
        """
        Count total number of parameters of the Neural Network model.
        """
        num_params = 0
        for i in range(1, self.N_layers):
            num_params += self.neural_net_topology[i-1] * self.neural_net_topology[i] # number of params in weights
            num_params += self.neural_net_topology[i] # number of params in biases
        print("Total # of Parameters = %d" % num_params)
        return num_params
    
    def performNeuralNetworkPrediction(self, dataset, dropout_keep_prob=1.0, is_training=False):
        """
        Perform Neural Network Prediction on a given dataset.
        :param dataset: dataset on which prediction will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: output tensor (in output layer)
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            hidden_drop = dataset
            for i in range(1, self.N_layers):
                affine_intermediate_result = tf.layers.dense(hidden_drop, self.neural_net_topology[i], name="ffnn_dense_"+str(i))
                
                if (self.is_using_batch_normalization):
                    activation_func_input = tf.layers.batch_normalization(affine_intermediate_result, training=is_training, name="ffnn_bn_"+str(i))
                else:
                    activation_func_input = affine_intermediate_result
                
                if (self.neural_net_activation_func_list[i] == 'identity'):
                    activation_func_output = activation_func_input
                elif (self.neural_net_activation_func_list[i] == 'tanh'):
                    activation_func_output = tf.nn.tanh(activation_func_input)
                elif (self.neural_net_activation_func_list[i] == 'relu'):
                    activation_func_output = tf.nn.relu(activation_func_input)
                else:
                    sys.exit('Unrecognized activation function: ' + self.neural_net_activation_func_list[i])
                
                if (i < self.N_layers - 1):     # Hidden Layer
                    hidden = activation_func_output
                    hidden_drop = tf.nn.dropout(hidden, dropout_keep_prob)
                    # the commented line below is BUGGY, sometimes causing NaNs (for large networks???).
#                    hidden_drop = tf.layers.dropout(inputs=hidden, rate=dropout_keep_prob, training=is_training, name="ffnn_do_"+str(i))
                else:                           # Output Layer (no Dropout here!)
                    output = activation_func_output
            return output