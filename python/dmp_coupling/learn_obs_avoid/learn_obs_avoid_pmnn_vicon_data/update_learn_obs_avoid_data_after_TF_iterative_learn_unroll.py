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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../neural_nets/feedforward/pmnn/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from TauSystem import *
from CanonicalSystemDiscrete import *
from CartesianCoordTransformer import *
from CartesianCoordDMP import *
from PMNN import *
from utilities import *


## Baseline Primitive Loading
dmp_baseline_params = loadObj('dmp_baseline_params_obs_avoid.pkl')
# end of Baseline Primitive Loading
ccdmp_baseline_params = dmp_baseline_params["cart_coord"][0]
dmp_out_dirpath = '../../../../data/dmp_coupling/learn_obs_avoid/static_obs/learned_prims_params/position/prim1/'

task_servo_rate = 300.0
dt = 1.0/task_servo_rate
model_size = 25
tau = 0.5
canonical_order = 2

tau_sys = TauSystem(dt, tau)
canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
ccdmp = CartesianCoordDMP(model_size, canonical_sys_discr, SCHAAL_LOCAL_COORD_FRAME)

ccdmp.saveParamsFromDict(dir_path=dmp_out_dirpath, cart_coord_dmp_params=ccdmp_baseline_params, 
                         file_name_weights="w", 
                         file_name_A_learn="A_learn", 
                         file_name_mean_start_position="start_global", 
                         file_name_mean_goal_position="goal_global", 
                         file_name_mean_tau="mean_tau", 
                         file_name_canonical_system_order="canonical_sys_order", 
                         file_name_mean_start_position_global="start_global", 
                         file_name_mean_goal_position_global="goal_global", 
                         file_name_mean_start_position_local="start_local", 
                         file_name_mean_goal_position_local="goal_local", 
                         file_name_ctraj_local_coordinate_frame_selection="ctraj_local_coordinate_frame_selection", 
                         file_name_ctraj_hmg_transform_local_to_global_matrix="T_local_to_global_H", 
                         file_name_ctraj_hmg_transform_global_to_local_matrix="T_global_to_local_H")
np.savetxt(dmp_out_dirpath + 'tau', np.array([ccdmp_baseline_params['unroll_tau']]))

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