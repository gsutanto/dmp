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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_coupling/learn_obs_avoid/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_coupling/learn_obs_avoid/vicon/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_coupling/learn_obs_avoid/learn_obs_avoid_pmnn_vicon_data/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../dmp_coupling/utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../utilities/'))
from DMPTrajectory import *
from DMPState import *
from TauSystem import *
from DMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from CartesianCoordTransformer import *
from CartesianCoordDMP import *
from TCLearnObsAvoidFeatureParameter import *
from TransformCouplingLearnObsAvoid import *
from unrollLearnedObsAvoidViconTraj import *
from DataStacking import *
from utilities import *
from vicon_obs_avoid_utils import *

def ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test(amd_clmc_dmp_home_dir_path="../../../../../../", 
                                                      canonical_order=2, unroll_ctraj_save_dir_path="", 
                                                      unroll_ctraj_save_filename=""):
    task_servo_rate = 300.0
    dt = 1.0/task_servo_rate
    dmp_basis_funcs_size = 25
    PMNN_input_size = 17
    PMNN_output_size = 3
    pmnn_model_parent_dir_path = amd_clmc_dmp_home_dir_path + "python/dmp_coupling/learn_obs_avoid/tf/models/"
    loa_data_dir_path = amd_clmc_dmp_home_dir_path + "data/dmp_coupling/learn_obs_avoid/"
    loa_data_prim_dir_path = loa_data_dir_path + "static_obs/learned_prims_params/position/prim1/"
    pmnn_model_path = loa_data_dir_path + "static_obs/neural_nets/pmnn/cpp_models/prim1/"
    data_global_coord_filepath = amd_clmc_dmp_home_dir_path + "python/dmp_coupling/learn_obs_avoid/learn_obs_avoid_pmnn_vicon_data/data_multi_demo_vicon_static_global_coord.pkl"
    pmnn_name = 'my_PMNN_obs_avoid_fb'
    ctraj_local_coordinate_frame_selection = GSUTANTO_LOCAL_COORD_FRAME
    is_using_scaling = [False] * PMNN_output_size # NOT using scaling on CartCoordDMP for now...
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    
    max_num_trajs_per_setting = 2
    selected_obs_avoid_setting_numbers = [26, 118]
    
    tau_sys = TauSystem(dt, MIN_TAU)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    loa_parameters = TCLearnObsAvoidFeatureParameter(PMNN_input_size,
                                                     dmp_basis_funcs_size, PMNN_output_size,
                                                     pmnn_model_parent_dir_path, 
                                                     pmnn_model_path,
                                                     PMNN_MODEL, pmnn_name)
    tcloa = TransformCouplingLearnObsAvoid(loa_parameters, tau_sys)
    transform_couplers_list = [tcloa]
    cart_coord_dmp = CartesianCoordDMP(dmp_basis_funcs_size, canonical_sys_discr, 
                                       ctraj_local_coordinate_frame_selection,
                                       transform_couplers_list)
    cart_coord_dmp.setScalingUsage(is_using_scaling)
    cart_coord_dmp.loadParamsCartCoordDMP(loa_data_prim_dir_path,
                                          "w",
                                          "A_learn",
                                          "start_global",
                                          "goal_global",
                                          "tau",
                                          "start_local",
                                          "goal_local",
                                          "T_local_to_global_H",
                                          "T_global_to_local_H")
    ccdmp_baseline_params = cart_coord_dmp.getParamsCartCoordDMPasDict()
    
    if (os.path.isfile(data_global_coord_filepath)):
        data_global_coord = loadObj(data_global_coord_filepath)
    else:
        data_global_coord = prepareDemoDatasetLOAVicon(task_servo_rate, amd_clmc_dmp_home_dir_path)
        
    N_unroll = 1 + (len(selected_obs_avoid_setting_numbers) * max_num_trajs_per_setting)
    list_global_traj_unroll = [None] * N_unroll
    list_global_position_traj_unroll = [None] * N_unroll
    
    ## without obstacle:
    [_,
     _,
     list_global_traj_unroll[0]] = unrollLearnedObsAvoidViconTraj(data_global_coord["obs_avoid"][1][selected_obs_avoid_setting_numbers[0]][0],
                                                                  data_global_coord["obs_avoid"][0][selected_obs_avoid_setting_numbers[0]],
                                                                  data_global_coord["dt"],
                                                                  ccdmp_baseline_params,
                                                                  cart_coord_dmp, False)
    list_global_position_traj_unroll[0] = list_global_traj_unroll[0].X.T
    
    ## with obstacle:
    list_idx = 1
    for ns in selected_obs_avoid_setting_numbers:
        for nd in range(max_num_trajs_per_setting):
            [_,
             _,
             list_global_traj_unroll[list_idx]] = unrollLearnedObsAvoidViconTraj(data_global_coord["obs_avoid"][1][ns][nd],
                                                                                 data_global_coord["obs_avoid"][0][ns],
                                                                                 data_global_coord["dt"],
                                                                                 ccdmp_baseline_params,
                                                                                 cart_coord_dmp, True)
            list_global_position_traj_unroll[list_idx] = list_global_traj_unroll[list_idx].X.T
            
            list_idx = list_idx + 1
    
    global_position_traj_unroll_w_timing = np.vstack(list_global_position_traj_unroll)
    
    if (os.path.isdir(unroll_ctraj_save_dir_path)):
        np.savetxt(unroll_ctraj_save_dir_path + "/" + unroll_ctraj_save_filename, global_position_traj_unroll_w_timing)
    
    return list_global_traj_unroll, global_position_traj_unroll_w_timing

if __name__ == "__main__":
    list_global_traj_unroll, global_position_traj_unroll_w_timing = ct_loa_so_sb_multi_demo_vicon_PMNN_unrolling_test("../../../../../../", 2)