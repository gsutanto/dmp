#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Wed Apr 10 15:00:00 2018

@author: gsutanto
"""

import scipy.io as sio
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from PMNN import *
from utilities import *
from copy import deepcopy


class PMNNv2(PMNN, object):
  """
    Same as the original PMNN, except that each output dimension AND
    each phase node has a separate network.
    In other words, there are D_output * N_phases separate networks in this
    model.
    """

  def __init__(self,
               name,
               D_input,
               regular_hidden_layer_topology,
               regular_hidden_layer_activation_func_list,
               N_phaseLWR_kernels,
               D_output,
               path='',
               is_using_phase_kernel_modulation=True,
               is_predicting_only=False):
    self.N_phases = N_phaseLWR_kernels
    super(PMNNv2,
          self).__init__(name, D_input, regular_hidden_layer_topology,
                         regular_hidden_layer_activation_func_list,
                         N_phaseLWR_kernels, D_output, path,
                         is_using_phase_kernel_modulation, is_predicting_only)
    print('This is PMNNv2.')

  def defineNeuralNetworkModel(self):
    """
        Define the Neural Network model.
        """
    assert (self.is_predicting_only == False), (
        'Needs to be inside a TensorFlow session, therefore '
        'self.is_predicting_only must be False!')

    num_params = 0
    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      for dim_out in range(self.D_output):
        dim_out_ID = 'o' + str(dim_out)
        if (
            i < self.N_layers - 1
        ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
          for phase_num in range(self.N_phases):
            phase_ID = 'p' + str(phase_num)
            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
            if (i < self.N_layers - 2):  # Regular Hidden Layers
              current_layer_dim_size = self.neural_net_topology[i]
            elif (i == self.N_layers -
                  2):  # Final Hidden Layer with Phase Gating/Modulation
              current_layer_dim_size = 1
            with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=False):
              weights = tf.get_variable(
                  'weights',
                  [self.neural_net_topology[i - 1], current_layer_dim_size],
                  initializer=tf.random_normal_initializer(0.0, 1e-14, seed=38))
              weights_dim = weights.get_shape().as_list()
              num_params += weights_dim[0] * weights_dim[1]
              # biases = tf.get_variable('biases', [current_layer_dim_size], initializer=tf.constant_initializer(0.0))
              biases = tf.get_variable(
                  'biases', [current_layer_dim_size],
                  initializer=tf.random_normal_initializer(0.0, 1e-14, seed=38))
              num_params += biases.get_shape().as_list()[0]
        else:  # Output Layer
          layer_dim_ID = layer_name + '_' + dim_out_ID
          current_layer_dim_size = 1
          with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=False):
            weights = tf.get_variable(
                'weights',
                [self.neural_net_topology[i - 1], current_layer_dim_size],
                initializer=tf.random_normal_initializer(0.0, 1e-14, seed=38))
            weights_dim = weights.get_shape().as_list()
            num_params += weights_dim[0] * weights_dim[1]
            # Output Layer does NOT have biases!!!
    num_params /= self.D_output
    print('Total # of Parameters = %d' % num_params)
    return num_params

  def performNeuralNetworkPrediction(self,
                                     dataset,
                                     normalized_phase_kernels,
                                     dropout_keep_prob=1.0):
    """
        Perform Neural Network Prediction on a given dataset.
        :param dataset: dataset on which prediction will be performed
        :param normalized_phase_kernels: normalized phase-based Gaussian kernels
        activation associated with the dataset
        :param dropout_keep_prob: probability of keeping a node (instead of
        dropping it; 1.0 means no drop-out)
        :param layer_num: layer number, at which the prediction would like to be
        made (default is at output (self.N_layers-1))
        :return: output tensor (in output layer)
        """
    layer_num = self.N_layers - 1  # Output Layer (default)
    assert layer_num < self.N_layers, 'layer_num must be < N_layers'
    hidden_dim = [
        [None for p in range(self.N_phases)] for d in range(self.D_output)
    ]
    hidden_drop_dim = [
        [dataset for p in range(self.N_phases)] for d in range(self.D_output)
    ]
    phase_dim = [None for d in range(self.D_output)]
    output_dim = [None for d in range(self.D_output)]
    for i in range(1, layer_num + 1):
      layer_name = self.getLayerName(i)

      for dim_out in range(self.D_output):
        dim_out_ID = 'o' + str(dim_out)
        if (
            i < self.N_layers - 1
        ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
          for phase_num in range(self.N_phases):
            phase_ID = 'p' + str(phase_num)
            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
            if (i < self.N_layers - 2):  # Regular Hidden Layers
              current_layer_dim_size = self.neural_net_topology[i]
            elif (i == self.N_layers -
                  2):  # Final Hidden Layer with Phase Gating/Modulation
              current_layer_dim_size = 1
            if (self.is_predicting_only == False):
              with tf.variable_scope(
                  self.name + '_' + layer_dim_ID, reuse=True):
                weights = tf.get_variable(
                    'weights',
                    [self.neural_net_topology[i - 1], current_layer_dim_size])
                biases = tf.get_variable('biases', [current_layer_dim_size])
                if (i < self.N_layers - 2):  # Regular Hidden Layer
                  affine_intermediate_result = tf.matmul(
                      hidden_drop_dim[dim_out][phase_num], weights) + biases
                  if (self.neural_net_activation_func_list[i] == 'identity'):
                    activation_func_output = affine_intermediate_result
                  elif (self.neural_net_activation_func_list[i] == 'tanh'):
                    activation_func_output = tf.nn.tanh(
                        affine_intermediate_result)
                  elif (self.neural_net_activation_func_list[i] == 'relu'):
                    activation_func_output = tf.nn.relu(
                        affine_intermediate_result)
                  else:
                    sys.exit('Unrecognized activation function: ' +
                             self.neural_net_activation_func_list[i])

                  hidden_dim[dim_out][phase_num] = activation_func_output
                  hidden_drop_dim[dim_out][phase_num] = tf.nn.dropout(
                      hidden_dim[dim_out][phase_num], dropout_keep_prob)
                elif (i == self.N_layers -
                      2):  # Final Hidden Layer with Phase Gating/Modulation
                  if (self.is_using_phase_kernel_modulation):
                    hidden_dim[dim_out][phase_num] = tf.reshape(
                        normalized_phase_kernels[:, phase_num],
                        [normalized_phase_kernels.get_shape().as_list()[0], 1
                        ]) * (
                            tf.matmul(hidden_drop_dim[dim_out][phase_num],
                                      weights) + biases)
                    hidden_drop_dim[dim_out][phase_num] = hidden_dim[dim_out][
                        phase_num]  # no dropout
                  else:  # if NOT using phase kernel modulation:
                    hidden_dim[dim_out][phase_num] = tf.nn.tanh(
                        tf.matmul(hidden_drop_dim[dim_out][phase_num], weights)
                        + biases)
                    hidden_drop_dim[dim_out][phase_num] = tf.nn.dropout(
                        hidden_dim[dim_out][phase_num], dropout_keep_prob)
            else:  # if (self.is_predicting_only == True)
              weights = self.model_params[self.name + '_' + layer_dim_ID +
                                          '_weights']
              biases = self.model_params[self.name + '_' + layer_dim_ID +
                                         '_biases']
              if (i < self.N_layers - 2):  # Regular Hidden Layer
                affine_intermediate_result = np.matmul(
                    hidden_drop_dim[dim_out][phase_num], weights) + biases
                if (self.neural_net_activation_func_list[i] == 'identity'):
                  activation_func_output = affine_intermediate_result
                elif (self.neural_net_activation_func_list[i] == 'tanh'):
                  activation_func_output = np.tanh(affine_intermediate_result)
                elif (self.neural_net_activation_func_list[i] == 'relu'):
                  activation_func_output = affine_intermediate_result * (
                      affine_intermediate_result > 0)
                else:
                  sys.exit('Unrecognized activation function: ' +
                           self.neural_net_activation_func_list[i])

                hidden_dim[dim_out][phase_num] = activation_func_output
                hidden_drop_dim[dim_out][phase_num] = hidden_dim[dim_out][
                    phase_num]
              elif (i == self.N_layers -
                    2):  # Final Hidden Layer with Phase Gating/Modulation
                if (self.is_using_phase_kernel_modulation):
                  hidden_dim[dim_out][
                      phase_num] = normalized_phase_kernels[:, phase_num].reshape(
                          normalized_phase_kernels.shape[0], 1) * (
                              np.matmul(hidden_drop_dim[dim_out][phase_num],
                                        weights) + biases)
                else:  # if NOT using phase kernel modulation:
                  hidden_dim[dim_out][phase_num] = np.tanh(
                      np.matmul(hidden_drop_dim[dim_out][phase_num], weights) +
                      biases)
                hidden_drop_dim[dim_out][phase_num] = hidden_dim[dim_out][
                    phase_num]
        else:  # Output Layer
          layer_dim_ID = layer_name + '_' + dim_out_ID
          current_layer_dim_size = 1
          if (self.is_predicting_only == False):
            with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
              weights = tf.get_variable(
                  'weights',
                  [self.neural_net_topology[i - 1], current_layer_dim_size])
              phase_dim[dim_out] = tf.concat(hidden_drop_dim[dim_out], 1)
              output_current_dim = tf.matmul(phase_dim[dim_out], weights)
              output_dim[dim_out] = output_current_dim
          else:  # if (self.is_predicting_only == True)
            weights = self.model_params[self.name + '_' + layer_dim_ID +
                                        '_weights']
            phase_dim[dim_out] = np.hstack(hidden_drop_dim[dim_out])
            output_current_dim = np.matmul(phase_dim[dim_out], weights)
            output_dim[dim_out] = output_current_dim
    if (self.is_predicting_only == False):
      output = tf.concat(output_dim, 1)
    else:
      output = np.hstack(output_dim)
    return output

  def computeLoss(self, prediction, ground_truth, beta):
    assert (self.is_predicting_only == False), (
        'Needs to be inside a TensorFlow session, therefore '
        'self.is_predicting_only must be False!')

    # Create an operation that calculates L2 prediction loss.
    pred_l2_loss = tf.nn.l2_loss(
        prediction - ground_truth, name='my_pred_L2_loss')

    # Create an operation that calculates L2 regularization loss.
    reg_l2_loss = 0
    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      for dim_out in range(self.D_output):
        dim_out_ID = 'o' + str(dim_out)
        if (
            i < self.N_layers - 1
        ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
          for phase_num in range(self.N_phases):
            phase_ID = 'p' + str(phase_num)
            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
            if (i < self.N_layers - 2):  # Regular Hidden Layers
              current_layer_dim_size = self.neural_net_topology[i]
            elif (i == self.N_layers -
                  2):  # Final Hidden Layer with Phase Gating/Modulation
              current_layer_dim_size = 1
            with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
              weights = tf.get_variable(
                  'weights',
                  [self.neural_net_topology[i - 1], current_layer_dim_size])
              reg_l2_loss = reg_l2_loss + tf.nn.l2_loss(weights)
              biases = tf.get_variable('biases', [current_layer_dim_size])
              reg_l2_loss = reg_l2_loss + tf.nn.l2_loss(biases)
        else:  # Output Layer
          layer_dim_ID = layer_name + '_' + dim_out_ID
          current_layer_dim_size = 1
          with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
            weights = tf.get_variable(
                'weights',
                [self.neural_net_topology[i - 1], current_layer_dim_size])
            reg_l2_loss = reg_l2_loss + tf.nn.l2_loss(weights)
            # Output Layer does NOT have biases!!!

    loss = tf.reduce_mean(
        pred_l2_loss, name='my_pred_L2_loss_mean') + (
            beta * reg_l2_loss)

    return loss

  def computeLossPerDimOut(self, prediction, ground_truth, beta, dim_out):
    assert (self.is_predicting_only == False), (
        'Needs to be inside a TensorFlow session, therefore '
        'self.is_predicting_only must be False!')

    # Create an operation that calculates L2 prediction loss.
    pred_l2_loss_dim = tf.nn.l2_loss(prediction[:, dim_out] -
                                     ground_truth[:, dim_out])

    # Create an operation that calculates L2 regularization loss.
    reg_l2_loss_dim = 0
    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      dim_out_ID = 'o' + str(dim_out)
      if (
          i < self.N_layers - 1
      ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
        for phase_num in range(self.N_phases):
          phase_ID = 'p' + str(phase_num)
          layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
          if (i < self.N_layers - 2):  # Regular Hidden Layers
            current_layer_dim_size = self.neural_net_topology[i]
          elif (i == self.N_layers -
                2):  # Final Hidden Layer with Phase Gating/Modulation
            current_layer_dim_size = 1
          with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
            weights = tf.get_variable(
                'weights',
                [self.neural_net_topology[i - 1], current_layer_dim_size])
            reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(weights)
            biases = tf.get_variable('biases', [current_layer_dim_size])
            reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(biases)
      else:  # Output Layer
        layer_dim_ID = layer_name + '_' + dim_out_ID
        current_layer_dim_size = 1
        with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
          weights = tf.get_variable(
              'weights',
              [self.neural_net_topology[i - 1], current_layer_dim_size])
          reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(weights)
          # Output Layer does NOT have biases!!!

    loss_dim = tf.reduce_mean(pred_l2_loss_dim) + (beta * reg_l2_loss_dim)

    return loss_dim

  def computeWeightedLossPerDimOut(self, prediction, ground_truth, beta,
                                   dim_out, weight):
    assert (self.is_predicting_only == False), (
        'Needs to be inside a TensorFlow session, therefore '
        'self.is_predicting_only must be False!')

    # Create an operation that calculates L2 prediction loss.
    pred_l2_loss_dim = 0.5 * tf.reduce_sum(
        weight * tf.square(prediction[:, dim_out] - ground_truth[:, dim_out]))

    # Create an operation that calculates L2 regularization loss.
    reg_l2_loss_dim = 0
    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      dim_out_ID = 'o' + str(dim_out)
      if (
          i < self.N_layers - 1
      ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
        for phase_num in range(self.N_phases):
          phase_ID = 'p' + str(phase_num)
          layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
          if (i < self.N_layers - 2):  # Regular Hidden Layers
            current_layer_dim_size = self.neural_net_topology[i]
          elif (i == self.N_layers -
                2):  # Final Hidden Layer with Phase Gating/Modulation
            current_layer_dim_size = 1
          with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
            weights = tf.get_variable(
                'weights',
                [self.neural_net_topology[i - 1], current_layer_dim_size])
            reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(weights)
            biases = tf.get_variable('biases', [current_layer_dim_size])
            reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(biases)
      else:  # Output Layer
        layer_dim_ID = layer_name + '_' + dim_out_ID
        current_layer_dim_size = 1
        with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
          weights = tf.get_variable(
              'weights',
              [self.neural_net_topology[i - 1], current_layer_dim_size])
          reg_l2_loss_dim = reg_l2_loss_dim + tf.nn.l2_loss(weights)
          # Output Layer does NOT have biases!!!

    loss_dim = pred_l2_loss_dim + (beta * reg_l2_loss_dim)

    return loss_dim

  # Save model in MATLAB format.
  def saveNeuralNetworkToMATLABMatFile(self):
    """
        Save the Neural Network model into a MATLAB *.m file.
        """
    assert (self.is_predicting_only == False), (
        'Needs to be inside a TensorFlow session, therefore '
        'self.is_predicting_only must be False!')

    self.model_params = {}
    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      for dim_out in range(self.D_output):
        dim_out_ID = 'o' + str(dim_out)
        if (
            i < self.N_layers - 1
        ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
          for phase_num in range(self.N_phases):
            phase_ID = 'p' + str(phase_num)
            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
            if (i < self.N_layers - 2):  # Regular Hidden Layers
              current_layer_dim_size = self.neural_net_topology[i]
            elif (i == self.N_layers -
                  2):  # Final Hidden Layer with Phase Gating/Modulation
              current_layer_dim_size = 1
            with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
              weights = tf.get_variable(
                  'weights',
                  [self.neural_net_topology[i - 1], current_layer_dim_size])
              self.model_params[self.name + '_' + layer_dim_ID +
                                '_weights'] = weights.eval()
              biases = tf.get_variable('biases', [current_layer_dim_size])
              self.model_params[self.name + '_' + layer_dim_ID +
                                '_biases'] = biases.eval()
        else:  # Output Layer
          layer_dim_ID = layer_name + '_' + dim_out_ID
          current_layer_dim_size = 1
          with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
            weights = tf.get_variable(
                'weights',
                [self.neural_net_topology[i - 1], current_layer_dim_size])
            self.model_params[self.name + '_' + layer_dim_ID +
                              '_weights'] = weights.eval()
            # Output Layer does NOT have biases!!!

    return self.model_params

  # Load model from MATLAB format.
  def loadNeuralNetworkFromMATLABMatFile(self, filepath):
    """
        Load a Neural Network model from a MATLAB *.mat file. Functionally
        comparable to the defineNeuralNetworkModel() function.
        :param filepath: (relative) path in the directory structure specifying
        the location of the file to be loaded.
        """
    num_params = 0
    self.model_params = sio.loadmat(filepath, struct_as_record=True)
    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      for dim_out in range(self.D_output):
        dim_out_ID = 'o' + str(dim_out)
        if (
            i < self.N_layers - 1
        ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
          for phase_num in range(self.N_phases):
            phase_ID = 'p' + str(phase_num)
            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
            if (i < self.N_layers - 2):  # Regular Hidden Layers
              current_layer_dim_size = self.neural_net_topology[i]
            elif (i == self.N_layers -
                  2):  # Final Hidden Layer with Phase Gating/Modulation
              current_layer_dim_size = 1
            if (self.is_predicting_only == False):
              with tf.variable_scope(
                  self.name + '_' + layer_dim_ID, reuse=False):
                weights = tf.get_variable(
                    'weights',
                    initializer=self.model_params[self.name + '_' +
                                                  layer_dim_ID + '_weights'])
                weights_dim = weights.get_shape().as_list()
                num_params += weights_dim[0] * weights_dim[1]
                biases = tf.get_variable(
                    'biases',
                    initializer=self.model_params[self.name + '_' +
                                                  layer_dim_ID +
                                                  '_biases'][0, :])
                num_params += biases.get_shape().as_list()[0]
            else:
              weights = self.model_params[self.name + '_' + layer_dim_ID +
                                          '_weights']
              weights_dim = list(weights.shape)
              num_params += weights_dim[0] * weights_dim[1]
              biases = self.model_params[self.name + '_' + layer_dim_ID +
                                         '_biases']
              num_params += list(biases.shape)[0]
        else:  # Output Layer
          layer_dim_ID = layer_name + '_' + dim_out_ID
          current_layer_dim_size = 1
          if (self.is_predicting_only == False):
            with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=False):
              weights = tf.get_variable(
                  'weights',
                  initializer=self.model_params[self.name + '_' + layer_dim_ID +
                                                '_weights'])
              weights_dim = weights.get_shape().as_list()
              num_params += weights_dim[0] * weights_dim[1]
          else:
            weights = self.model_params[self.name + '_' + layer_dim_ID +
                                        '_weights']
            weights_dim = list(weights.shape)
            num_params += weights_dim[0] * weights_dim[1]
          # Output Layer does NOT have biases!!!
    num_params /= self.D_output
    print('Total # of Parameters = %d' % num_params)
    return num_params

  # Save model in *.txt format.
  def saveNeuralNetworkToTextFiles(self, dirpath):
    """
        Save the Neural Network model into text (*.txt) files in directory
        specified by dirpath.
        """
    recreateDir(dirpath)

    for dim_out in range(self.D_output):
      os.makedirs(dirpath + '/o' + str(dim_out))
      for phase_num in range(self.N_phases):
        os.makedirs(dirpath + '/o' + str(dim_out) + '/p' + str(phase_num))

    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      for dim_out in range(self.D_output):
        dim_out_ID = 'o' + str(dim_out)
        if (
            i < self.N_layers - 1
        ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
          for phase_num in range(self.N_phases):
            phase_ID = 'p' + str(phase_num)
            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
            if (i < self.N_layers - 2):  # Regular Hidden Layers
              current_layer_dim_size = self.neural_net_topology[i]
            elif (i == self.N_layers -
                  2):  # Final Hidden Layer with Phase Gating/Modulation
              current_layer_dim_size = 1
            if (self.is_predicting_only == False):
              with tf.variable_scope(
                  self.name + '_' + layer_dim_ID, reuse=True):
                weights_tf = tf.get_variable(
                    'weights',
                    [self.neural_net_topology[i - 1], current_layer_dim_size])
                weights = weights_tf.eval()
                biases_tf = tf.get_variable('biases', [current_layer_dim_size])
                biases = biases_tf.eval()
            else:
              weights = self.model_params[self.name + '_' + layer_dim_ID +
                                          '_weights']
              biases = self.model_params[self.name + '_' + layer_dim_ID +
                                         '_biases']
            np.savetxt(
                dirpath + '/' + dim_out_ID + '/' + phase_ID + '/w' + str(i - 1),
                weights)
            np.savetxt(
                dirpath + '/' + dim_out_ID + '/' + phase_ID + '/b' + str(i - 1),
                biases)
        else:  # Output Layer
          layer_dim_ID = layer_name + '_' + dim_out_ID
          current_layer_dim_size = 1
          if (self.is_predicting_only == False):
            with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=True):
              weights_tf = tf.get_variable(
                  'weights',
                  [self.neural_net_topology[i - 1], current_layer_dim_size])
              weights = weights_tf.eval()
          else:
            weights = self.model_params[self.name + '_' + layer_dim_ID +
                                        '_weights']
          np.savetxt(dirpath + '/' + dim_out_ID + '/w' + str(i - 1), weights)
          # Output Layer does NOT have biases!!!

    return None

  # Load model from *.txt files format.
  def loadNeuralNetworkFromTextFiles(self, dirpath):
    """
        Load a Neural Network model from text (*.txt) files in directory
        specified by dirpath.
        Functionally comparable to the defineNeuralNetworkModel() function.
        :param dirpath: (relative) path in the directory structure specifying
        the location of the files to be loaded.
        """
    num_params = 0
    self.model_params = {}
    for i in range(1, self.N_layers):
      layer_name = self.getLayerName(i)

      for dim_out in range(self.D_output):
        dim_out_ID = 'o' + str(dim_out)
        if (
            i < self.N_layers - 1
        ):  # Hidden Layers (including the Final Hidden Layer with Phase Gating/Modulation)
          for phase_num in range(self.N_phases):
            phase_ID = 'p' + str(phase_num)
            layer_dim_ID = layer_name + '_' + dim_out_ID + '_' + phase_ID
            if (i < self.N_layers - 2):  # Regular Hidden Layers
              current_layer_dim_size = self.neural_net_topology[i]
            elif (i == self.N_layers -
                  2):  # Final Hidden Layer with Phase Gating/Modulation
              current_layer_dim_size = 1
            weights_temp = np.loadtxt(dirpath + '/' + dim_out_ID + '/' +
                                      phase_ID + '/w' + str(i - 1))
            if (len(weights_temp.shape) == 1):
              weights_temp = weights_temp.reshape(weights_temp.shape[0], 1)
            self.model_params[self.name + '_' + layer_dim_ID +
                              '_weights'] = weights_temp
            biases_temp = np.loadtxt(dirpath + '/' + dim_out_ID + '/' +
                                     phase_ID + '/b' + str(i - 1))
            self.model_params[self.name + '_' + layer_dim_ID +
                              '_biases'] = biases_temp
            if (self.is_predicting_only == False):
              with tf.variable_scope(
                  self.name + '_' + layer_dim_ID, reuse=False):
                weights = tf.get_variable(
                    'weights',
                    initializer=self.model_params[self.name + '_' +
                                                  layer_dim_ID + '_weights'])
                weights_dim = weights.get_shape().as_list()
                num_params += weights_dim[0] * weights_dim[1]
                biases = tf.get_variable(
                    'biases',
                    initializer=self.model_params[self.name + '_' +
                                                  layer_dim_ID +
                                                  '_biases'][0, :])
                num_params += biases.get_shape().as_list()[0]
            else:
              weights = self.model_params[self.name + '_' + layer_dim_ID +
                                          '_weights']
              weights_dim = list(weights.shape)
              num_params += weights_dim[0] * weights_dim[1]
              biases = self.model_params[self.name + '_' + layer_dim_ID +
                                         '_biases']
              num_params += list(biases.shape)[0]
        else:  # Output Layer
          layer_dim_ID = layer_name + '_' + dim_out_ID
          current_layer_dim_size = 1
          weights_temp = np.loadtxt(dirpath + '/' + dim_out_ID + '/w' +
                                    str(i - 1))
          if (len(weights_temp.shape) == 1):
            weights_temp = weights_temp.reshape(weights_temp.shape[0], 1)
          self.model_params[self.name + '_' + layer_dim_ID +
                            '_weights'] = weights_temp
          if (self.is_predicting_only == False):
            with tf.variable_scope(self.name + '_' + layer_dim_ID, reuse=False):
              weights = tf.get_variable(
                  'weights',
                  initializer=self.model_params[self.name + '_' + layer_dim_ID +
                                                '_weights'])
              weights_dim = weights.get_shape().as_list()
              num_params += weights_dim[0] * weights_dim[1]
          else:
            weights = self.model_params[self.name + '_' + layer_dim_ID +
                                        '_weights']
            weights_dim = list(weights.shape)
            num_params += weights_dim[0] * weights_dim[1]
          # Output Layer does NOT have biases!!!
    num_params /= self.D_output
    print('Total # of Parameters = %d' % num_params)
    return num_params
