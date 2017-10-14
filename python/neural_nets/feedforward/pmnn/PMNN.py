#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:00:00 2017

@author: gsutanto
"""

import scipy.io as sio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from FeedForwardNeuralNetwork import *
from copy import deepcopy

class PMNN(FeedForwardNeuralNetwork):
    """
    Class for special feed-forward neural network,
    the final hidden layer is gated/modulated by phase-LWR.
    For each dimension of the output, there is one dedicated neural network.
    All neural networks (i.e. for all output dimensions) are optimized together.
    """
    
    def __init__(self, name, D_input, 
                 regular_hidden_layer_topology, regular_hidden_layer_activation_func_list, 
                 N_phaseLWR_kernels, D_output, 
                 filepath="", is_using_phase_kernel_modulation=True):
        self.name = name
        
        self.neural_net_topology = [D_input] + regular_hidden_layer_topology + [N_phaseLWR_kernels, D_output]
        print "PMNN Topology:"
        print self.neural_net_topology
        
        self.D_output = D_output
        self.N_layers = len(self.neural_net_topology)
        
        if (regular_hidden_layer_activation_func_list == []):
            self.neural_net_activation_func_list = ['identity'] * self.N_layers
        else:
            assert (len(regular_hidden_layer_activation_func_list) == len(regular_hidden_layer_topology)), "len(regular_hidden_layer_activation_func_list) must be == len(regular_hidden_layer_topology)"
            self.neural_net_activation_func_list = ['identity'] + regular_hidden_layer_activation_func_list + ['identity', 'identity']
        # First Layer (Input Layer) always uses 'identity' activation function (and it does NOT matter actually; this is mainly for the sake of layer-indexing consistency...).
        assert (len(self.neural_net_activation_func_list) == self.N_layers), "len(self.neural_net_activation_func_list) must be == self.N_layers"
        print "Neural Network Activation Function List:"
        print self.neural_net_activation_func_list
        
        if (filepath == ""):
            self.num_params = self.defineNeuralNetworkModel()
        else:
            self.num_params = self.loadNeuralNetworkFromMATLABMatFile(filepath)
        
        self.is_using_phase_kernel_modulation = is_using_phase_kernel_modulation
    
    def getLayerName(self, layer_index):
        assert layer_index > 0, "layer_index must be > 0"
        assert layer_index < self.N_layers, "layer_index must be < N_layers"
        if (layer_index == self.N_layers - 1):  # Output Layer
            layer_name = 'output'
        elif (layer_index == self.N_layers - 2):  # Final Hidden Layer with Phase LWR Gating/Modulation
            layer_name = 'phaseLWR'
        else:  # Hidden Layer
            layer_name = 'hidden' + str(layer_index)
        return layer_name
    
    def defineNeuralNetworkModel(self):
        """
        Define the Neural Network model.
        """
        num_params = 0
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
            
            for dim_out in range(self.D_output):
                layer_dim_ID = layer_name + '_' + str(dim_out)
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation)
                    current_layer_dim_size = self.neural_net_topology[i]
                else: # Output Layer
                    current_layer_dim_size = 1
                with tf.variable_scope(self.name+'_'+layer_dim_ID, reuse=False):
                    weights = tf.get_variable('weights', [self.neural_net_topology[i-1], current_layer_dim_size], initializer=tf.random_normal_initializer(0.0, 1e-14, seed=38))
                    weights_dim = weights.get_shape().as_list()
                    num_params += weights_dim[0] * weights_dim[1]
                    if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation); Output Layer does NOT have biases!!!
                        # biases = tf.get_variable('biases', [current_layer_dim_size], initializer=tf.constant_initializer(0.0))
                        biases = tf.get_variable('biases', [current_layer_dim_size], initializer=tf.random_normal_initializer(0.0, 1e-14, seed=38))
                        num_params += biases.get_shape().as_list()[0]
        num_params /= self.D_output
        print("Total # of Parameters = %d" % num_params)
        return num_params
    
    def performNeuralNetworkPrediction(self, dataset, normalized_phase_kernels, dropout_keep_prob=1.0, layer_num=-1):
        """
        Perform Neural Network Prediction on a given dataset.
        :param dataset: dataset on which prediction will be performed
        :param normalized_phase_kernels: normalized phase-based Gaussian kernels activation associated with the dataset
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :param layer_num: layer number, at which the prediction would like to be made (default is at output (self.N_layers-1))
        :return: output tensor (in output layer)
        """
        if (layer_num == -1):
            layer_num = self.N_layers-1 # Output Layer (default)
        assert layer_num < self.N_layers, "layer_num must be < N_layers"
        hidden_dim = [None] * self.D_output
        hidden_drop_dim = [dataset] * self.D_output
        output_dim = list()
        for i in range(1, layer_num+1):
            layer_name = self.getLayerName(i)
    
            for dim_out in range(self.D_output):
                layer_dim_ID = layer_name + '_' + str(dim_out)
                
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation)
                    current_layer_dim_size = self.neural_net_topology[i]
                else: # Output Layer
                    current_layer_dim_size = 1
                
                with tf.variable_scope(self.name+'_'+layer_dim_ID, reuse=True):
                    weights = tf.get_variable('weights', [self.neural_net_topology[i-1], current_layer_dim_size])
                    if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation); Output Layer does NOT have biases!!!
                        biases = tf.get_variable('biases', [current_layer_dim_size])
                    
                    if (i < self.N_layers - 2):  # Regular Hidden Layer
                        affine_intermediate_result = tf.matmul(hidden_drop_dim[dim_out], weights) + biases
                        if (self.neural_net_activation_func_list[i] == 'identity'):
                            activation_func_output = affine_intermediate_result
                        elif (self.neural_net_activation_func_list[i] == 'tanh'):
                            activation_func_output = tf.nn.tanh(affine_intermediate_result)
                        elif (self.neural_net_activation_func_list[i] == 'relu'):
                            activation_func_output = tf.nn.relu(affine_intermediate_result)
                        else:
                            sys.exit('Unrecognized activation function: ' + self.neural_net_activation_func_list[i])
                        
                        hidden_dim[dim_out] = activation_func_output
                        hidden_drop_dim[dim_out] = tf.nn.dropout(hidden_dim[dim_out], dropout_keep_prob)
                    elif (i == self.N_layers - 2): # Final Hidden Layer with Phase LWR Gating/Modulation
                        if (self.is_using_phase_kernel_modulation == True):
                            hidden_dim[dim_out] = normalized_phase_kernels * (tf.matmul(hidden_drop_dim[dim_out], weights) + biases)
                            hidden_drop_dim[dim_out] = hidden_dim[dim_out] # no dropout
                        else:   # if NOT using phase kernel modulation:
                            hidden_dim[dim_out] = tf.nn.tanh(tf.matmul(hidden_drop_dim[dim_out], weights) + biases)
                            hidden_drop_dim[dim_out] = tf.nn.dropout(hidden_dim[dim_out], dropout_keep_prob)
                    else: # Output Layer
                        output_current_dim = tf.matmul(hidden_drop_dim[dim_out], weights)
                        output_dim.append(output_current_dim)
        if (layer_num == self.N_layers-1):
            output = tf.concat(1, output_dim)
            return output
        else:
            return hidden_dim
    
    # def performNeuralNetworkTraining(self, prediction, ground_truth, initial_learning_rate, beta, N_steps):
    def performNeuralNetworkTraining(self, prediction, ground_truth, initial_learning_rate, beta):
        """
        Perform Neural Network Training (joint, across all output dimensions).
        :param prediction: prediction made by current model on a dataset
        :param ground_truth: the ground truth of the corresponding dataset
        :param initial_learning_rate: initial learning rate
        :param beta: L2 regularization constant
        """
        # Create an operation that calculates L2 prediction loss.
        pred_l2_loss = tf.nn.l2_loss(prediction - ground_truth, name='my_pred_L2_loss')
    
        # Create an operation that calculates L2 regularization loss.
        reg_l2_loss = 0
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            for dim_out in range(self.D_output):
                layer_dim_ID = layer_name + '_' + str(dim_out)
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation)
                    current_layer_dim_size = self.neural_net_topology[i]
                else: # Output Layer
                    current_layer_dim_size = 1
                with tf.variable_scope(self.name+'_'+layer_dim_ID, reuse=True):
                    weights = tf.get_variable('weights', [self.neural_net_topology[i-1], current_layer_dim_size])
                    reg_l2_loss = reg_l2_loss + tf.nn.l2_loss(weights)
                    if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation); Output Layer does NOT have biases!!!
                        biases = tf.get_variable('biases', [current_layer_dim_size])
                        reg_l2_loss = reg_l2_loss + tf.nn.l2_loss(biases)
    
        loss = tf.reduce_mean(pred_l2_loss, name='my_pred_L2_loss_mean') + (beta * reg_l2_loss)
    
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
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
        train_op_dim = tf.train.RMSPropOptimizer(initial_learning_rate).minimize(loss, global_step=global_step)
        
        # return train_op, loss, learning_rate
        return train_op, loss
    
    # def performNeuralNetworkTraining(self, prediction, ground_truth, initial_learning_rate, beta, N_steps):
    def performNeuralNetworkTrainingPerDimOut(self, prediction, ground_truth, initial_learning_rate, beta, dim_out):
        """
        Perform Neural Network Training (per output dimension).
        :param prediction: prediction made by current model on a dataset
        :param ground_truth: the ground truth of the corresponding dataset
        :param initial_learning_rate: initial learning rate
        :param beta: L2 regularization constant
        :param dim_out: output dimension being considered
        """
        # Create an operation that calculates L2 prediction loss.
        pred_l2_loss_dim = tf.nn.l2_loss(prediction[:,dim_out] - ground_truth[:,dim_out])
        
        # Create an operation that calculates L2 regularization loss.
        reg_l2_loss_dim = 0
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
            layer_dim_ID = layer_name + '_' + str(dim_out)
            if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation)
                current_layer_dim_size = self.neural_net_topology[i]
            else: # Output Layer
                current_layer_dim_size = 1
            with tf.variable_scope(self.name+'_'+layer_dim_ID, reuse=True):
                weights = tf.get_variable('weights', [self.neural_net_topology[i-1], current_layer_dim_size])
                reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(weights)
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation); Output Layer does NOT have biases!!!
                    biases = tf.get_variable('biases', [current_layer_dim_size])
                    reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(biases)
    
            loss_dim = tf.reduce_mean(pred_l2_loss_dim) + (beta * reg_l2_loss_dim)
        
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Exponentially-decaying learning rate:
        # learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, N_steps, 0.1)
        # Create the gradient descent optimizer with the given learning rate.
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # train_op = tf.train.MomentumOptimizer(learning_rate, momentum=learning_rate/4.0, use_nesterov=True).minimize(loss, global_step=global_step)
        # train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        # train_op_dim = tf.train.AdagradOptimizer(initial_learning_rate).minimize(loss_dim, global_step=global_step)
        # train_op_dim = tf.train.AdadeltaOptimizer().minimize(loss_dim, global_step=global_step)
        train_op_dim = tf.train.RMSPropOptimizer(initial_learning_rate).minimize(loss_dim, global_step=global_step)
        
        return train_op_dim, loss_dim
    
    # def performNeuralNetworkTraining(self, prediction, ground_truth, initial_learning_rate, beta, N_steps):
    def performNeuralNetworkWeightedTrainingPerDimOut(self, prediction, ground_truth, initial_learning_rate, beta, dim_out, weight):
        """
        Perform Neural Network Training (per output dimension).
        :param prediction: prediction made by current model on a dataset
        :param ground_truth: the ground truth of the corresponding dataset
        :param initial_learning_rate: initial learning rate
        :param beta: L2 regularization constant
        :param dim_out: output dimension being considered
        :param weight: weight vector corresponding to the prediction/dataset
        """
        # Create an operation that calculates L2 prediction loss.
        pred_l2_loss_dim = 0.5 * tf.reduce_sum(weight * tf.square(prediction[:,dim_out] - ground_truth[:,dim_out]))
        
        # Create an operation that calculates L2 regularization loss.
        reg_l2_loss_dim = 0
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
            layer_dim_ID = layer_name + '_' + str(dim_out)
            if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation)
                current_layer_dim_size = self.neural_net_topology[i]
            else: # Output Layer
                current_layer_dim_size = 1
            with tf.variable_scope(self.name+'_'+layer_dim_ID, reuse=True):
                weights = tf.get_variable('weights', [self.neural_net_topology[i-1], current_layer_dim_size])
                reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(weights)
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation); Output Layer does NOT have biases!!!
                    biases = tf.get_variable('biases', [current_layer_dim_size])
                    reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(biases)
    
            loss_dim = pred_l2_loss_dim + (beta * reg_l2_loss_dim)
        
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Exponentially-decaying learning rate:
        # learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, N_steps, 0.1)
        # Create the gradient descent optimizer with the given learning rate.
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # train_op = tf.train.MomentumOptimizer(learning_rate, momentum=learning_rate/4.0, use_nesterov=True).minimize(loss, global_step=global_step)
        # train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        # train_op_dim = tf.train.AdagradOptimizer(initial_learning_rate).minimize(loss_dim, global_step=global_step)
        # train_op_dim = tf.train.AdadeltaOptimizer().minimize(loss_dim, global_step=global_step)
        train_op_dim = tf.train.RMSPropOptimizer(initial_learning_rate).minimize(loss_dim, global_step=global_step)
        
        return train_op_dim, loss_dim

    # Save model.
    def saveNeuralNetworkToMATLABMatFile(self):
        """
        Save the Neural Network model into a MATLAB *.m file.
        """
        model_params={}
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            for dim_out in range(self.D_output):
                layer_dim_ID = layer_name + '_' + str(dim_out)
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation)
                    current_layer_dim_size = self.neural_net_topology[i]
                else: # Output Layer
                    current_layer_dim_size = 1
                with tf.variable_scope(self.name+'_'+layer_dim_ID, reuse=True):
                    weights = tf.get_variable('weights', [self.neural_net_topology[i-1], current_layer_dim_size])
                    model_params[self.name+'_'+layer_dim_ID+"_weights"] = weights.eval()
                    if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation); Output Layer does NOT have biases!!!
                        biases = tf.get_variable('biases', [current_layer_dim_size])
                        model_params[self.name+'_'+layer_dim_ID+"_biases"] = biases.eval()
    
        return model_params

    # Load model.
    def loadNeuralNetworkFromMATLABMatFile(self, filepath):
        """
        Load a Neural Network model from a MATLAB *.m file. Functionally comparable to the defineNeuralNetworkModel() function.
        :param filepath: (relative) path in the directory structure specifying the location of the file to be loaded.
        """
        num_params = 0
        model_params = sio.loadmat(filepath, struct_as_record=True)
        for i in range(1, self.N_layers):
            layer_name = self.getLayerName(i)
    
            for dim_out in range(self.D_output):
                layer_dim_ID = layer_name + '_' + str(dim_out)
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation)
                    current_layer_dim_size = self.neural_net_topology[i]
                else: # Output Layer
                    current_layer_dim_size = 1
                with tf.variable_scope(self.name+'_'+layer_dim_ID, reuse=False):
                    weights = tf.get_variable('weights', initializer=model_params[self.name+'_'+layer_dim_ID+"_weights"])
                    weights_dim = weights.get_shape().as_list()
                    num_params += weights_dim[0] * weights_dim[1]
                    if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase LWR Gating/Modulation); Output Layer does NOT have biases!!!
                        biases = tf.get_variable('biases', initializer=model_params[self.name+'_'+layer_dim_ID+"_biases"][0,:])
                        num_params += biases.get_shape().as_list()[0]
        num_params /= self.D_output
        print("Total # of Parameters = %d" % num_params)
        return num_params
