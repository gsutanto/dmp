#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 22:00:00 2017

@author: gsutanto
@comment: version 2: Using tf.layers library and Batch Normalization
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from FeedForwardNeuralNetworkV2 import *
from copy import deepcopy

class AutoEncoderV2(FeedForwardNeuralNetworkV2):
    """
    Class for special feed-forward neural network, the AutoEncoder.
    Decoder's topology is the mirror/reverse copy of the encoder's topology.
    Also, the last two layers (final hidden layer and output layer) 
    are dimension-specific (version 2).
    """
    
    def __init__(self, name, D_input, 
                 encoder_hidden_layer_topology, encoder_hidden_layer_activation_func_list, 
                 D_latent, is_using_batch_normalization=True):
        self.name = name
        
        self.neural_net_topology = [D_input] + encoder_hidden_layer_topology + [D_latent] + list(reversed(encoder_hidden_layer_topology)) + [D_input]
        print self.name + " AutoEncoder Topology:"
        print self.neural_net_topology
        
        self.D_output = D_input
        self.N_layers = len(self.neural_net_topology)
        
        if (encoder_hidden_layer_activation_func_list == []):
            self.neural_net_activation_func_list = ['identity'] * self.N_layers
        else:
            assert (len(encoder_hidden_layer_activation_func_list) == len(encoder_hidden_layer_topology)), "len(encoder_hidden_layer_activation_func_list) must be == len(encoder_hidden_layer_topology)"
            self.neural_net_activation_func_list = ['identity'] + encoder_hidden_layer_activation_func_list + ['identity'] + list(reversed(encoder_hidden_layer_activation_func_list)) + ['identity']
        # First Layer (Input Layer) always uses 'identity' activation function (and it does NOT matter actually; this is mainly for the sake of layer-indexing consistency...).
        assert (len(self.neural_net_activation_func_list) == self.N_layers), "len(self.neural_net_activation_func_list) must be == self.N_layers"
        print "Neural Network Activation Function List:"
        print self.neural_net_activation_func_list
        
        self.num_params = self.countNeuralNetworkModelNumParams()
        
        self.is_using_batch_normalization = is_using_batch_normalization
    
    def encode(self, input_dataset, dropout_keep_prob=1.0, is_training=False):
        """
        Perform encoding on a given input dataset.
        :param input_dataset: input dataset on which encoding will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: latent tensor (in latent layer)
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            hidden_drop = input_dataset
            for i in range(1, ((self.N_layers - 1) / 2) + 1):
                affine_intermediate_result = tf.layers.dense(hidden_drop, self.neural_net_topology[i], name="ae_dense_"+str(i))
                
                if (self.is_using_batch_normalization):
                    activation_func_input = tf.layers.batch_normalization(affine_intermediate_result, training=is_training, name="ae_bn_"+str(i))
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
                
                if (i < ((self.N_layers - 1) / 2)): # Encoder's Hidden Layer
                    hidden = activation_func_output
                    hidden_drop = tf.nn.dropout(hidden, dropout_keep_prob)
                    # the commented line below is BUGGY, sometimes causing NaNs (for large networks???).
#                    hidden_drop = tf.layers.dropout(inputs=hidden, rate=dropout_keep_prob, training=is_training, name="ae_do_"+str(i))
                elif (i == (self.N_layers - 1) / 2): # Latent Layer (no Dropout here!)
                    latent = activation_func_output
            return latent
    
    def decode(self, latent_dataset, dropout_keep_prob=1.0, is_training=False):
        """
        Perform decoding on a given latent dataset.
        :param latent_dataset: latent dataset on which decoding will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: output tensor (in output layer)
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            hidden_drop = latent_dataset
            for i in range(((self.N_layers - 1) / 2) + 1, self.N_layers):
                affine_intermediate_result = tf.layers.dense(hidden_drop, self.neural_net_topology[i], name="ae_dense_"+str(i))
                
                if (self.is_using_batch_normalization):
                    activation_func_input = tf.layers.batch_normalization(affine_intermediate_result, training=is_training, name="ae_bn_"+str(i))
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
                
                if (i < (self.N_layers - 1)): # Decoder's Hidden Layer
                    hidden = activation_func_output
                    hidden_drop = tf.nn.dropout(hidden, dropout_keep_prob)
                    # the commented line below is BUGGY, sometimes causing NaNs (for large networks???).
#                    hidden_drop = tf.layers.dropout(inputs=hidden, rate=dropout_keep_prob, training=is_training, name="ae_do_"+str(i))
                elif (i == self.N_layers - 1): # Output Layer (no Dropout here!)
                    output = activation_func_output
            return output
    
    def performNeuralNetworkPrediction(self, dataset, dropout_keep_prob=1.0, is_training=False):
        """
        Perform Neural Network Prediction: reconstruction of a given input dataset.
        :param dataset: input dataset on which prediction/reconstruction will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: reconstructed tensor (in output layer)
        """
        latent_dataset = self.encode(dataset, dropout_keep_prob, is_training)
        reconstructed_dataset = self.decode(latent_dataset, dropout_keep_prob, is_training)
        return reconstructed_dataset
