#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../neural_nets/feedforward/pmnn/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../neural_nets/feedforward/rpmnn/'))
from PMNN import *
from RPMNN import *

PMNN_MODEL = 0
RPMNN_MODEL = 1

class TCLearnObsAvoidFeatureParameter:
    'Class for a group/bundle of states of obstacles.'
    
    def __init__(self, model_dim_inputs,
                 model_dim_phase_mod_kernels=25, model_dim_outputs=3,
                 model_parent_dir_path='../tf/models/',
                 model_path=None, 
                 model=PMNN_MODEL, name=""):
        self.name = name
        self.model = model
        self.D_input = model_dim_inputs
        if (self.model == PMNN_MODEL):
            if (model_path is None):
                reinit_selection_idx = list(np.loadtxt(model_parent_dir_path+'reinit_selection_idx.txt', dtype=np.int, ndmin=1))
                TF_max_train_iters = np.loadtxt(model_parent_dir_path+'TF_max_train_iters.txt', dtype=np.int, ndmin=0)
                prim_no = 1
                self.pmnn_model_path = model_parent_dir_path + 'prim_' + str(prim_no) + '_params_reinit_' + str(reinit_selection_idx[prim_no-1]) + ('_step_%07d.mat' % TF_max_train_iters)
            else:
                self.pmnn_model_path = model_path
            self.pmnn_regular_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
            self.pmnn_regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1))
            self.pmnn = PMNN(self.name, model_dim_inputs, 
                             self.pmnn_regular_NN_hidden_layer_topology, 
                             self.pmnn_regular_NN_hidden_layer_activation_func_list, 
                             model_dim_phase_mod_kernels, model_dim_outputs, 
                             self.pmnn_model_path, True, True)
        elif (self.model == RPMNN_MODEL):
            if (model_path is None):
                reinit_selection_idx = list(np.loadtxt(model_parent_dir_path+'reinit_selection_idx.txt', dtype=np.int, ndmin=1))
                TF_max_train_iters = np.loadtxt(model_parent_dir_path+'TF_max_train_iters.txt', dtype=np.int, ndmin=0)
                prim_no = 1
                self.rpmnn_model_path = model_parent_dir_path + 'diff_Ct_dataset_prim_' + str(prim_no) + '_params_reinit_' + str(reinit_selection_idx[prim_no-1]) + ('_step_%07d.mat' % TF_max_train_iters)
            else:
                self.rpmnn_model_path = model_path
            self.rpmnn_regular_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
            self.rpmnn_regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1))
            self.rpmnn = RPMNN(self.name, model_dim_inputs, 
                               self.rpmnn_regular_NN_hidden_layer_topology, 
                               self.rpmnn_regular_NN_hidden_layer_activation_func_list, 
                               model_dim_phase_mod_kernels, model_dim_outputs, 
                               self.rpmnn_model_path, True, True)
    
    def isValid(self):
        return True