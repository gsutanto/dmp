#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 22:00:00 2017

@author: gsutanto
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from FeedForwardNeuralNetwork import *
from copy import deepcopy

class AutoEncoder(FeedForwardNeuralNetwork):
    """
    Class for special feed-forward neural network, the AutoEncoder.
    Decoder's topology is the mirror/reverse copy of the encoder's topology.
    Also, the last two layers (final hidden layer and output layer) are dimension-specific.
    """
    
    def __init__(self, name, D_input, encoder_hidden_layer_topology, D_latent, filepath=""):
        self.name = name
        self.neural_net_topology = [D_input] + encoder_hidden_layer_topology + [D_latent] + list(reversed(encoder_hidden_layer_topology)) + [D_input]
        print "AutoEncoder Topology:"
        print self.neural_net_topology
        self.D_output = D_input
        self.N_layers = len(self.neural_net_topology)
        if (filepath == ""):
            self.num_params = self.defineNeuralNetworkModel()
        else:
            self.num_params = self.loadNeuralNetworkFromMATLABMatFile(filepath)
    
    def getLayerName(self, layer_index):
        assert layer_index > 0, "layer_index must be > 0"
        assert layer_index < self.N_layers, "layer_index must be < N_layers"
        if (layer_index == self.N_layers - 1): # Output Layer
            layer_name = 'output'
        elif (layer_index == (self.N_layers - 1) / 2): # Latent Layer
            layer_name = 'latent'
        elif ((layer_index >= 1) and (layer_index < (self.N_layers - 1) / 2)): # Encoder's Hidden Layer
            layer_name = 'encoder_hidden' + str(layer_index)
        elif ((layer_index > (self.N_layers - 1) / 2) and (layer_index < self.N_layers - 1)): # Decoder's Hidden Layer
            layer_name = 'decoder_hidden' + str(layer_index - ((self.N_layers - 1) / 2))
        return layer_name
    
    def encode(self, input_dataset, dropout_keep_prob=1.0):
        """
        Perform encoding on a given input dataset.
        :param input_dataset: input dataset on which encoding will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: latent tensor (in latent layer)
        """
        hidden_drop = input_dataset
        for i in range(1, ((self.N_layers - 1) / 2) + 1):
            layer_name = self.getLayerName(i)
    
            with tf.variable_scope(self.name+'_'+layer_name, reuse=True):
                weights = tf.get_variable('weights', [self.neural_net_topology[i - 1], self.neural_net_topology[i]])
                biases = tf.get_variable('biases', [self.neural_net_topology[i]])
                if (i < ((self.N_layers - 1) / 2)): # Encoder's Hidden Layer
                    # hidden = tf.nn.relu(tf.matmul(hidden_drop, weights) + biases)
                    hidden = tf.nn.tanh(tf.matmul(hidden_drop, weights) + biases)
                    hidden_drop = tf.nn.dropout(hidden, dropout_keep_prob)
                elif (i == (self.N_layers - 1) / 2): # Latent Layer
                    latent = tf.matmul(hidden_drop, weights) + biases
        return latent
    
    def decode(self, latent_dataset, dropout_keep_prob=1.0):
        """
        Perform decoding on a given latent dataset.
        :param latent_dataset: latent dataset on which decoding will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: output tensor (in output layer)
        """
        hidden_drop = latent_dataset
        for i in range(((self.N_layers - 1) / 2) + 1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            with tf.variable_scope(self.name+'_'+layer_name, reuse=True):
                weights = tf.get_variable('weights', [self.neural_net_topology[i - 1], self.neural_net_topology[i]])
                biases = tf.get_variable('biases', [self.neural_net_topology[i]])
                if (i < (self.N_layers - 1)): # Decoder's Hidden Layer
                    # hidden = tf.nn.relu(tf.matmul(hidden_drop, weights) + biases)
                    hidden = tf.nn.tanh(tf.matmul(hidden_drop, weights) + biases)
                    hidden_drop = tf.nn.dropout(hidden, dropout_keep_prob)
                elif (i == self.N_layers - 1): # Output Layer
                    output = tf.matmul(hidden_drop, weights) + biases
        return output
    
    def performNeuralNetworkPrediction(self, dataset, dropout_keep_prob=1.0):
        """
        Perform Neural Network Prediction: reconstruction of a given input dataset.
        :param dataset: input dataset on which prediction/reconstruction will be performed
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: reconstructed tensor (in output layer)
        """
        latent_dataset = self.encode(dataset, dropout_keep_prob)
        reconstructed_dataset = self.decode(latent_dataset, dropout_keep_prob)
        return reconstructed_dataset
