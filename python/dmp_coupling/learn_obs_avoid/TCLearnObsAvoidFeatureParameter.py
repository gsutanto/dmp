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
from PMNN import *

PMNN_MODEL = 0

class TCLearnObsAvoidFeatureParameter:
    'Class for a group/bundle of states of obstacles.'
    
    def __init__(self, model_parent_dir_path='../tf/models/', model=PMNN_MODEL, name=""):
        self.name = name
        self.model = model
        if (model == PMNN_MODEL):
            reinit_selection_idx = list(np.loadtxt(model_parent_dir_path+'reinit_selection_idx.txt', dtype=np.int, ndmin=1))
            TF_max_train_iters = np.loadtxt(model_parent_dir_path+'TF_max_train_iters.txt', dtype=np.int, ndmin=0)
            prim_no = 1
            self.pmnn_model_filepath = model_parent_dir_path + 'prim_' + str(prim_no) + '_params_reinit_' + str(reinit_selection_idx[prim_no-1]) + ('_step_%07d.mat' % TF_max_train_iters)
            self.pmnn_regular_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
            self.pmnn_regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1))
            self.pmnn = PMNN(NN_name, D_input, 
                             self.pmnn_regular_NN_hidden_layer_topology, 
                             self.pmnn_regular_NN_hidden_layer_activation_func_list, 
                             N_phaseLWR_kernels, D_output, 
                             self.pmnn_model_filepath, True, True)
    
    def isValid(self):
        return True