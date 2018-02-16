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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/learn_obs_avoid/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from DMPTrajectory import *
from DMPState import *
from TauSystem import *
from DMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from CartesianCoordTransformer import *
from CartesianCoordDMP import *
from TCLearnObsAvoidFeatureParameter import *
from TransformCouplingLearnObsAvoid import *
from convertDemoToSupervisedObsAvoidFbDataset import *
from unrollLearnedObsAvoidViconTraj import *
from DataStacking import *
from utilities import *

## Demo Dataset Loading
data_global_coord = loadObj('data_multi_demo_vicon_static_global_coord.pkl')
# end of Demo Dataset Loading

## Baseline Primitive Loading
dmp_baseline_params = loadObj('dmp_baseline_params_obs_avoid.pkl')
# end of Baseline Primitive Loading

ccdmp_baseline_params = dmp_baseline_params["cart_coord"][0]

## Supervised Obstacle Avoidance Feedback Dataset Loading
dataset_Ct_obs_avoid = loadObj('dataset_Ct_obs_avoid.pkl')
# end of Supervised Obstacle Avoidance Feedback Dataset Loading

D_input = 17
D_output = 3
rpmnn_model_parent_dir_path = '../tf/models/'
# rpmnn_model_file_path = None
rpmnn_model_file_path = '../tf/models/iterative_learn_unroll/recur_Ct_dataset_prim_1_params_step_0000440.mat'
#rpmnn_model_file_path = '../tf/models/recur_Ct_dataset_prim_1_params_reinit_2_step_0500000.mat'
rpmnn_name = 'my_RPMNN_obs_avoid_fb'

dmp_basis_funcs_size = 25
canonical_order = 2
ctraj_local_coordinate_frame_selection = GSUTANTO_LOCAL_COORD_FRAME
is_using_scaling = [False] * D_output # NOT using scaling on CartCoordDMP for now...
                                        
tau_sys = TauSystem(data_global_coord["dt"], MIN_TAU)
canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
loa_parameters = TCLearnObsAvoidFeatureParameter(D_input,
                                                 dmp_basis_funcs_size, D_output,
                                                 rpmnn_model_parent_dir_path, 
                                                 rpmnn_model_file_path,
                                                 RPMNN_MODEL, rpmnn_name)
tcloa = TransformCouplingLearnObsAvoid(loa_parameters, tau_sys)
transform_couplers_list = [tcloa]
cart_coord_dmp = CartesianCoordDMP(dmp_basis_funcs_size, canonical_sys_discr, 
                                   ctraj_local_coordinate_frame_selection,
                                   transform_couplers_list)
cart_coord_dmp.setScalingUsage(is_using_scaling)
cart_coord_dmp.setParams(ccdmp_baseline_params['W'], ccdmp_baseline_params['A_learn'])

N_settings = len(data_global_coord["obs_avoid"][0])
prim_no = 0 # There is only one (1) primitive here.

unroll_dataset_Ct_obs_avoid = {}
unroll_dataset_Ct_obs_avoid["sub_X"] = [[None] * N_settings]
unroll_dataset_Ct_obs_avoid["sub_Ct_target"] = [[None] * N_settings]
global_traj_unroll = [[None] * N_settings]

for ns in range(N_settings):
    N_demos = len(data_global_coord["obs_avoid"][1][ns])
    
    # the index 0 before ns seems unnecessary, but this is just for the sake of generality, if we have multiple primitives
    unroll_dataset_Ct_obs_avoid["sub_X"][prim_no][ns] = [None] * N_demos
    unroll_dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns] = [None] * N_demos
    global_traj_unroll[prim_no][ns] = [None] * N_demos
    
    for nd in range(N_demos):
        print ('Setting #' + str(ns+1) + '/' + str(N_settings) + ', Demo #' + str(nd+1) + '/' + str(N_demos))
        [unroll_dataset_Ct_obs_avoid["sub_X"][prim_no][ns][nd],
         unroll_dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns][nd],
         global_traj_unroll[prim_no][ns][nd]] = unrollLearnedObsAvoidViconTraj(data_global_coord["obs_avoid"][1][ns][nd],
                                                                               data_global_coord["obs_avoid"][0][ns],
                                                                               data_global_coord["dt"],
                                                                               ccdmp_baseline_params,
                                                                               cart_coord_dmp)

saveObj(unroll_dataset_Ct_obs_avoid, 'unroll_dataset_Ct_obs_avoid_recur_Ct_dataset.pkl')
saveObj(global_traj_unroll, 'global_traj_unroll_recur_Ct_dataset.pkl')