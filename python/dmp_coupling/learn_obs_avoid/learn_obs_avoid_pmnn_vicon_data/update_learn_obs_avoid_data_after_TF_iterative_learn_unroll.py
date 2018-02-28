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
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../neural_nets/feedforward/pmnn/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from PMNN import *
from utilities import *


## Baseline Primitive Loading
dmp_baseline_params = loadObj('dmp_baseline_params_obs_avoid.pkl')
# end of Baseline Primitive Loading
ccdmp_baseline_params = dmp_baseline_params["cart_coord"][0]
dmp_out_dirpath = '../../../../data/dmp_coupling/learn_obs_avoid/static_obs/learned_prims_params/position/prim1/'

np.savetxt(dmp_out_dirpath + 'w', ccdmp_baseline_params['W'])
np.savetxt(dmp_out_dirpath + 'A_learn', ccdmp_baseline_params['A_learn'])
np.savetxt(dmp_out_dirpath + 'start_global', ccdmp_baseline_params['mean_start_global_position'])
np.savetxt(dmp_out_dirpath + 'goal_global', ccdmp_baseline_params['mean_goal_global_position'])
np.savetxt(dmp_out_dirpath + 'start_local', ccdmp_baseline_params['mean_start_local_position'])
np.savetxt(dmp_out_dirpath + 'goal_local', ccdmp_baseline_params['mean_goal_local_position'])
np.savetxt(dmp_out_dirpath + 'T_local_to_global_H', ccdmp_baseline_params['T_local_to_global_H'])
np.savetxt(dmp_out_dirpath + 'T_global_to_local_H', ccdmp_baseline_params['T_global_to_local_H'])
np.savetxt(dmp_out_dirpath + 'tau', np.array([ccdmp_baseline_params['unroll_tau']]))
np.savetxt(dmp_out_dirpath + 'canonical_sys_order', np.array([ccdmp_baseline_params['canonical_order']]))
np.savetxt(dmp_out_dirpath + 'ctraj_local_coordinate_frame_selection', np.array([ccdmp_baseline_params['ctraj_local_coordinate_frame_selection']]))

pmnn_name = 'my_PMNN_obs_avoid_fb'
D_input = 17
model_parent_dir_path = '../tf/models/'
regular_NN_hidden_layer_topology = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_topology.txt', dtype=np.int, ndmin=1))
regular_NN_hidden_layer_activation_func_list = list(np.loadtxt(model_parent_dir_path+'regular_NN_hidden_layer_activation_func_list.txt', dtype=np.str, ndmin=1))
N_phaseLWR_kernels = 25
D_output = 3
#init_model_param_filepath = model_parent_dir_path + 'iterative_learn_unroll/prim_1_params_step_0044300.mat'
init_model_param_filepath = model_parent_dir_path + 'iterative_learn_unroll/prim_1_params_step_0005500.mat'
pmnn_model_out_dirpath = '../../../../data/dmp_coupling/learn_obs_avoid/static_obs/neural_nets/pmnn/cpp_models/prim1/'

pmnn = PMNN(pmnn_name, D_input, 
            regular_NN_hidden_layer_topology, regular_NN_hidden_layer_activation_func_list, 
            N_phaseLWR_kernels, D_output, init_model_param_filepath, True, True)
pmnn.saveNeuralNetworkToTextFiles(pmnn_model_out_dirpath)