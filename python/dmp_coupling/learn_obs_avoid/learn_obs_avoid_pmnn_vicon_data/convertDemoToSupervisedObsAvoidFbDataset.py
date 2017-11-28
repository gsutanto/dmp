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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/learn_obs_avoid/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_coupling/learn_obs_avoid/vicon/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from DMPState import *
from CartesianCoordTransformer import *
from learnCartPrimitiveMultiOnLocalCoord import *
from DataIO import *
from utilities import *
from vicon_obs_avoid_utils import *
from ObstacleStates import *
from TransformCouplingLearnObsAvoid import *
from TCLearnObsAvoidFeatureParameter import *

def convertDemoToSupervisedObsAvoidFbDataset():
    n_rfs = 25  # Number of basis functions used to represent the forcing term of DMP
    c_order = 2 # DMP is using 2nd order canonical system
    
    ## Demo Dataset Preparation
    
    data_global_coord = prepareDemoDatasetLOAVicon()
    
    saveObj(data_global_coord, 'data_multi_demo_vicon_static_global_coord.pkl')
    
    # end of Demo Dataset Preparation
    
    ## Baseline Primitive Learning
    
    print('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...')
    [ccdmp_baseline_params,
     ccdmp_baseline_unroll_global_traj,
     _,
     _,
     cart_coord_dmp] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord["baseline"], 
                                                           data_global_coord["dt"],
                                                           n_rfs, 
                                                           c_order)
    
    dmp_baseline_params = {}
    dmp_baseline_params["cart_coord"] = [ccdmp_baseline_params]
    
    saveObj(dmp_baseline_params, 'dmp_baseline_params_obs_avoid.pkl')
    saveObj(ccdmp_baseline_unroll_global_traj, 'ccdmp_baseline_unroll_global_traj.pkl')
    
    # end of Baseline Primitive Learning
    
    ## Conversion of Demonstration Dataset into Supervised Obstacle Avoidance Feedback Model Dataset
    
    endeff_cart_state_global = DMPState(np.zeros((3,1)))
    point_obstacles_cart_state_global = ObstacleStates(np.zeros((3,21)),
                                                       np.zeros((3,21)),
                                                       np.zeros((3,21)),
                                                       np.zeros((1,1)))
    loa_parameters = None
    tcloa = TransformCouplingLearnObsAvoid(loa_parameters, cart_coord_dmp.tau_sys, cart_coord_dmp,
                                           endeff_cart_state_global, point_obstacles_cart_state_global)
    
    N_settings = len(data_global_coord["obs_avoid"][0])
    
    dataset_Ct_obs_avoid = {}
    dataset_Ct_obs_avoid["sub_X"] = [[None] * N_settings]
    dataset_Ct_obs_avoid["sub_Ct_target"] = [[None] * N_settings]
    dataset_Ct_obs_avoid["sub_phase_PSI"] = [[None] * N_settings]
    dataset_Ct_obs_avoid["sub_phase_V"] = [[None] * N_settings]
    dataset_Ct_obs_avoid["sub_phase_X"] = [[None] * N_settings]
    dataset_Ct_obs_avoid["sub_normalized_phase_PSI_mult_phase_V"] = [[None] * N_settings]
    dataset_Ct_obs_avoid["sub_data_point_priority"] = [[None] * N_settings]
    dataset_Ct_obs_avoid["trial_idx_ranked_by_outlier_metric_w_exclusion"] = [[None] * N_settings]
    min_num_considered_demo = 0
    
    prim_no = 0 # There is only one (1) primitive here.
    
    for ns in range(N_settings):
        N_demos = len(data_global_coord["obs_avoid"][1][ns])
        
        # the index 0 before ns seems unnecessary, but this is just for the sake of generality, if we have multiple primitives
        dataset_Ct_obs_avoid["sub_X"][prim_no][ns] = [None] * N_demos
        dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns] = [None] * N_demos
        dataset_Ct_obs_avoid["sub_phase_PSI"][prim_no][ns] = [None] * N_demos
        dataset_Ct_obs_avoid["sub_phase_V"][prim_no][ns] = [None] * N_demos
        dataset_Ct_obs_avoid["sub_phase_X"][prim_no][ns] = [None] * N_demos
        dataset_Ct_obs_avoid["sub_normalized_phase_PSI_mult_phase_V"][prim_no][ns] = [None] * N_demos
        dataset_Ct_obs_avoid["sub_data_point_priority"][prim_no][ns] = [None] * N_demos
        dataset_Ct_obs_avoid["trial_idx_ranked_by_outlier_metric_w_exclusion"][prim_no][ns] = []
        
        for nd in range(N_demos):
            print ('Setting #' + str(ns+1) + '/' + str(N_settings) + ', Demo #' + str(nd+1) + '/' + str(N_demos))
            [dataset_Ct_obs_avoid["sub_X"][prim_no][ns][nd],
             dataset_Ct_obs_avoid["sub_Ct_target"][prim_no][ns][nd],
             dataset_Ct_obs_avoid["sub_phase_PSI"][prim_no][ns][nd],
             dataset_Ct_obs_avoid["sub_phase_V"][prim_no][ns][nd],
             dataset_Ct_obs_avoid["sub_phase_X"][prim_no][ns][nd],
             is_good_demo] = tcloa.computeSubFeatMatAndSubTargetCt(data_global_coord["obs_avoid"][1][ns][nd],
                                                                   data_global_coord["obs_avoid"][0][ns],
                                                                   data_global_coord["dt"],
                                                                   ccdmp_baseline_params,
                                                                   cart_coord_dmp)
            
            phase_V = dataset_Ct_obs_avoid["sub_phase_V"][prim_no][ns][nd]
            phase_PSI = dataset_Ct_obs_avoid["sub_phase_PSI"][prim_no][ns][nd]
            traj_length = phase_PSI.shape[1]
            normalized_phase_PSI_mult_phase_V = phase_PSI * np.matmul(np.ones((n_rfs, 1)), (phase_V * 1.0 / np.sum(phase_PSI, axis=0).reshape((1,traj_length))))
            dataset_Ct_obs_avoid["sub_normalized_phase_PSI_mult_phase_V"][prim_no][ns][nd] = normalized_phase_PSI_mult_phase_V
            dataset_Ct_obs_avoid["sub_data_point_priority"][prim_no][ns][nd] = range(traj_length,0,-1)
            
            if (is_good_demo):
                dataset_Ct_obs_avoid["trial_idx_ranked_by_outlier_metric_w_exclusion"][prim_no][ns].append(nd)
        
        if (ns == 0):
            min_num_considered_demo = len(dataset_Ct_obs_avoid["trial_idx_ranked_by_outlier_metric_w_exclusion"][prim_no][ns])
        else:
            min_num_considered_demo = min(min_num_considered_demo, len(dataset_Ct_obs_avoid["trial_idx_ranked_by_outlier_metric_w_exclusion"][prim_no][ns]))
    
    print ('Minimum # of Considered Demonstrations = ' + str(min_num_considered_demo))
    saveObj(dataset_Ct_obs_avoid, 'dataset_Ct_obs_avoid.pkl')
    
    # end of Conversion of Demonstration Dataset into Supervised Obstacle Avoidance Feedback Model Dataset
    
    return data_global_coord, dmp_baseline_params, ccdmp_baseline_unroll_global_traj, dataset_Ct_obs_avoid, min_num_considered_demo