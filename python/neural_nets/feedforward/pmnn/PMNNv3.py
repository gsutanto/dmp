#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:00:00 2018

@author: gsutanto
"""

import scipy.io as sio
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from PMNNv2 import *
from utilities import *
from copy import deepcopy

class PMNNv3(PMNNv2, object):
    """
    Same as the PMNNv2 but with Batch Normalization layers.
    """
    
    def __init__(self, name, D_input, 
                 regular_hidden_layer_topology, regular_hidden_layer_activation_func_list, 
                 N_phaseLWR_kernels, D_output, regularization_const=0.0, 
                 path="", is_using_phase_kernel_modulation=True, is_predicting_only=False, 
                 is_using_batch_normalization=True):
        assert (is_predicting_only == False), 'PMNNv3 only supports is_predicting_only == False, i.e. execution inside a TensorFlow session.'
        
        self.name = name
        
        self.neural_net_topology = [D_input] + regular_hidden_layer_topology + [N_phaseLWR_kernels, D_output]
        print ("PMNN %s Topology:" % self.name)
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
        
        self.N_phases = N_phaseLWR_kernels
        
        self.is_predicting_only = is_predicting_only
        self.num_params = self.countNeuralNetworkModelNumParams()
        
        self.is_using_phase_kernel_modulation = is_using_phase_kernel_modulation
        
        self.is_using_batch_normalization = is_using_batch_normalization
        
        self.regularization_const = regularization_const
        
        print "This is PMNNv3."
    
    def countNeuralNetworkModelNumParams(self):
        """
        Count total number of parameters of the PMNNv3 model.
        This does NOT include Batch Normalization parameters (if used).
        """
        assert (self.is_predicting_only == False), 'Needs to be inside a TensorFlow session, therefore self.is_predicting_only must be False!'
        
        num_params = 0
        for i in range(1, self.N_layers):
            for dim_out in range(self.D_output):
                if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
                    for phase_num in range(self.N_phases):
                        if (i < self.N_layers - 2): # Regular Hidden Layers
                            current_layer_dim_size = self.neural_net_topology[i]
                        elif (i == self.N_layers - 2): # Final Hidden Layer with Phase Gating/Modulation
                            current_layer_dim_size = 1
                        num_params += self.neural_net_topology[i-1] * current_layer_dim_size # number of params in weights
                        num_params += current_layer_dim_size # number of params in biases
                else: # Output Layer
                    current_layer_dim_size = 1
                    num_params += self.neural_net_topology[i-1] * current_layer_dim_size
                    # Output Layer does NOT have biases!!!
        num_params /= self.D_output
        print("Total # of Parameters = %d" % num_params)
        return num_params
    
    def performNeuralNetworkPrediction(self, dataset, normalized_phase_kernels, dropout_keep_prob=1.0, is_training=False):
        """
        Perform Neural Network Prediction on a given dataset.
        :param dataset: dataset on which prediction will be performed
        :param normalized_phase_kernels: normalized phase-based Gaussian kernels activation associated with the dataset
        :param dropout_keep_prob: probability of keeping a node (instead of dropping it; 1.0 means no drop-out)
        :return: output tensor (in output layer)
        """
        assert (self.is_predicting_only == False), 'Needs to be inside a TensorFlow session, therefore self.is_predicting_only must be False!'
        
        layer_num = self.N_layers-1 # Output Layer (default)
        assert layer_num < self.N_layers, "layer_num must be < N_layers"
        hidden_dim = [[None for p in range(self.N_phases)] for d in range(self.D_output)]
        hidden_drop_dim = [[dataset for p in range(self.N_phases)] for d in range(self.D_output)]
        phase_dim = [None for d in range(self.D_output)]
        output_dim = [None for d in range(self.D_output)]
        for i in range(1, layer_num+1):
            layer_name = self.getLayerName(i)
            
            for dim_out in range(self.D_output):
                dim_out_ID = 'o' + str(dim_out)
                with tf.variable_scope(self.name+'_'+dim_out_ID, reuse=tf.AUTO_REUSE):
                    if (i < self.N_layers - 1): # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
                        for phase_num in range(self.N_phases):
                            phase_ID = 'p' + str(phase_num)
                            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
                            if (i < self.N_layers - 2): # Regular Hidden Layers
                                current_layer_dim_size = self.neural_net_topology[i]
                            elif (i == self.N_layers - 2): # Final Hidden Layer with Phase Gating/Modulation
                                current_layer_dim_size = 1
                            if (i < self.N_layers - 2):  # Regular Hidden Layer
                                affine_intermediate_result = tf.layers.dense(hidden_drop_dim[dim_out][phase_num], 
                                                                             current_layer_dim_size, 
                                                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization_const), 
                                                                             name=self.name+'_'+layer_dim_ID+'_dense_'+str(i))
                
                                if (self.is_using_batch_normalization):
                                    activation_func_input = tf.layers.batch_normalization(affine_intermediate_result, 
                                                                                          training=is_training, 
                                                                                          name=self.name+'_'+layer_dim_ID+'_bn_'+str(i))
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
                                
                                hidden_dim[dim_out][phase_num] = activation_func_output
                                hidden_drop_dim[dim_out][phase_num] = tf.nn.dropout(hidden_dim[dim_out][phase_num], dropout_keep_prob)
                            elif (i == self.N_layers - 2): # Final Hidden Layer with Phase Gating/Modulation
                                affine_intermediate_result = tf.layers.dense(hidden_drop_dim[dim_out][phase_num], 
                                                                             current_layer_dim_size, 
                                                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization_const), 
                                                                             name=self.name+'_'+layer_dim_ID+'_dense_'+str(i))
                
                                if (self.is_using_batch_normalization):
                                    activation_func_input = tf.layers.batch_normalization(affine_intermediate_result, training=is_training, name=self.name+'_'+layer_dim_ID+'_bn_'+str(i))
                                else:
                                    activation_func_input = affine_intermediate_result
                                
                                if (self.is_using_phase_kernel_modulation):
                                    hidden_dim[dim_out][phase_num] = tf.reshape(normalized_phase_kernels[:,phase_num], [tf.shape(normalized_phase_kernels)[0],1]) * activation_func_input
                                    hidden_drop_dim[dim_out][phase_num] = hidden_dim[dim_out][phase_num] # no dropout
                                else:   # if NOT using phase kernel modulation:
                                    hidden_dim[dim_out][phase_num] = tf.nn.tanh(activation_func_input)
                                    hidden_drop_dim[dim_out][phase_num] = tf.nn.dropout(hidden_dim[dim_out][phase_num], dropout_keep_prob)
                    else: # Output Layer
                        layer_dim_ID = layer_name + '_' + dim_out_ID
                        current_layer_dim_size = 1
                        phase_dim[dim_out] = tf.concat(hidden_drop_dim[dim_out], 1)
                        output_current_dim = tf.layers.dense(phase_dim[dim_out], 
                                                             current_layer_dim_size, 
                                                             use_bias=False, 
                                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization_const), 
                                                             name=self.name+'_'+layer_dim_ID+'_dense_'+str(i))
                        output_dim[dim_out] = output_current_dim
        output = tf.concat(output_dim, 1)
        return output
    
    def computeLoss(self, prediction, ground_truth):
        assert (self.is_predicting_only == False), 'Needs to be inside a TensorFlow session, therefore self.is_predicting_only must be False!'
        
        # Create an operation that calculates L2 prediction loss.
        pred_l2_loss = tf.nn.l2_loss(prediction - ground_truth, name='my_pmnnv3_pred_L2_loss')
    
        # Create an operation that calculates L2 regularization loss.
        total_reg_l2_loss = 0
        for dim_out in range(self.D_output):
            dim_out_ID = 'o' + str(dim_out)
            total_reg_l2_loss = total_reg_l2_loss + tf.losses.get_regularization_loss(scope=self.name+'_'+dim_out_ID)
        
        loss = tf.reduce_mean(pred_l2_loss, name='my_pmnnv3_pred_L2_loss_mean') + total_reg_l2_loss
        
        return loss
    
    def computeLossPerDimOut(self, prediction, ground_truth, dim_out):
        assert (self.is_predicting_only == False), 'Needs to be inside a TensorFlow session, therefore self.is_predicting_only must be False!'
        
        # Create an operation that calculates L2 prediction loss.
        pred_l2_loss_dim = tf.nn.l2_loss(prediction[:,dim_out] - ground_truth[:,dim_out])
        
        # Create an operation that calculates L2 regularization loss.
        dim_out_ID = 'o' + str(dim_out)
        total_reg_l2_loss_dim = tf.losses.get_regularization_loss(scope=self.name+'_'+dim_out_ID)
        
        loss_dim = tf.reduce_mean(pred_l2_loss_dim) + total_reg_l2_loss_dim
        
        return loss_dim
    
    def computeWeightedLossPerDimOut(self, prediction, ground_truth, dim_out, weight):
        assert (self.is_predicting_only == False), 'Needs to be inside a TensorFlow session, therefore self.is_predicting_only must be False!'
        
        # Create an operation that calculates L2 prediction loss.
        pred_l2_loss_dim = 0.5 * tf.reduce_sum(weight * tf.square(prediction[:,dim_out] - ground_truth[:,dim_out]))
        
        # Create an operation that calculates L2 regularization loss.
        dim_out_ID = 'o' + str(dim_out)
        total_reg_l2_loss_dim = tf.losses.get_regularization_loss(scope=self.name+'_'+dim_out_ID)
            
        loss_dim = pred_l2_loss_dim + total_reg_l2_loss_dim
        
        return loss_dim
    
    def performNeuralNetworkWeightedTrainingPerDimOut(self, prediction, ground_truth, initial_learning_rate, dim_out, weight):
        """
        Perform Neural Network Training (per output dimension).
        :param prediction: prediction made by current model on a dataset
        :param ground_truth: the ground truth of the corresponding dataset
        :param initial_learning_rate: initial learning rate
        :param beta: L2 regularization constant
        :param dim_out: output dimension being considered
        :param weight: weight vector corresponding to the prediction/dataset
        """
        loss_dim = self.computeWeightedLossPerDimOut(prediction, ground_truth, dim_out, weight)
        
        # Create a variable to track the global step.
        global_step_dim = tf.Variable(0, name='global_step', trainable=False)
        with tf.variable_scope('PMNNv3dim'+str(dim_out)):
            opt_dim = tf.train.RMSPropOptimizer(learning_rate=initial_learning_rate)
            train_op_dim = opt_dim.minimize(loss_dim, global_step=global_step_dim)
        
        return train_op_dim, loss_dim