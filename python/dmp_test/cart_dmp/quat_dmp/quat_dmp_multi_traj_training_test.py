#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 01 16:00:00 2019

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/quat_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from TauSystem import *
from QuaternionDMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from QuaternionDMP import *
from utilities import *

plt.close('all')

def quat_dmp_multi_traj_training_test(dmp_home_dir_path="../../../../", 
                                      canonical_order=2, time_reproduce_max=2.0, 
                                      tau_reproduce=2.0, unroll_qtraj_save_dir_path="", 
                                      unroll_qtraj_save_filename="", 
                                      is_smoothing_training_traj_before_learning=False, 
                                      percentage_padding=None, percentage_smoothing_points=None, 
                                      smoothing_mode=None, smoothing_cutoff_frequency=None, 
                                      unroll_learned_params_save_dir_path="", 
                                      unroll_learned_params_save_filename="", 
                                      is_plotting=False):
    task_servo_rate = 300.0
    dt = 1.0/task_servo_rate
    model_size = 25
    tau = MIN_TAU
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    assert (time_reproduce_max >= 0.0)
    assert (tau_reproduce >= 0.0)
    
    tau_sys = TauSystem(dt, tau)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    quat_dmp = QuaternionDMP(model_size, canonical_sys_discr)
    tau = tau_reproduce
    
    unroll_qtraj_list = list()
    unroll_learned_params_list = list()
    
    for k in range(2):
        if (k == 0):
            sub_quat_dmp_training_path = "/data/cart_dmp/quat_dmp/multi_traj_training/"
        elif (k == 1):
#            sub_quat_dmp_training_path = "/data/dmp_coupling/learn_tactile_feedback/scraping_wo_tool/human_baseline/prim02/"
            # this one is a more challenging case, because Quaternions -Q and Q both represent the same orientation
            # (this is to see if the low-pass filtering on such Quaternion trajectory is done right; 
            #  if NOT low-pass filtering will just make things 
            #  (especially the QuaternionDMP fitting) so much worse and WRONG):
            sub_quat_dmp_training_path = "/data/cart_dmp/quat_dmp_unscrewing/prim02/"
        
        set_qtraj_input = quat_dmp.extractSetTrajectories(dmp_home_dir_path + sub_quat_dmp_training_path, 
                                                          start_column_idx=10, time_column_idx=0)
        
        [critical_states_learn, 
         W, mean_A_learn, mean_tau, 
         Ft, Fp, QgT, cX, cV, 
         PSI] = quat_dmp.learnFromSetTrajectories(set_qtraj_input, 
                                                  task_servo_rate, 
                                                  is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                                  percentage_padding=percentage_padding, 
                                                  percentage_smoothing_points=percentage_smoothing_points, 
                                                  smoothing_mode=smoothing_mode, 
                                                  smoothing_cutoff_frequency=smoothing_cutoff_frequency)
        
        ## Reproduce
        qdmp_unroll = quat_dmp.unroll(critical_states_learn, tau, time_reproduce_max, dt)
        unroll_qtraj_time = qdmp_unroll.time.T - dt
        unroll_qtraj_Q = qdmp_unroll.X.T
        unroll_qtraj = np.hstack([unroll_qtraj_time, unroll_qtraj_Q])
        unroll_qtraj_list.append(copy.deepcopy(unroll_qtraj))
        
        unroll_learned_params = np.zeros((model_size+1+2,5))
        unroll_learned_params[0:model_size,0:3] = W.T
        unroll_learned_params[model_size:model_size+1,0:3] = mean_A_learn.T
        unroll_learned_params[model_size,3] = mean_tau
        unroll_learned_params[model_size+1:model_size+2,0:4] = quat_dmp.mean_start_position.T
        unroll_learned_params[model_size+2:model_size+3,0:4] = quat_dmp.mean_goal_position.T
        unroll_learned_params_list.append(copy.deepcopy(unroll_learned_params))
    
    unroll_qtraj_concatenated = np.vstack(unroll_qtraj_list)
    unroll_learned_params_concatenated = np.vstack(unroll_learned_params_list)
    
    if (os.path.isdir(unroll_qtraj_save_dir_path)):
        np.savetxt(unroll_qtraj_save_dir_path + "/" + unroll_qtraj_save_filename, unroll_qtraj_concatenated)
    
    if (os.path.isdir(unroll_learned_params_save_dir_path)):
        np.savetxt(unroll_learned_params_save_dir_path + "/" + unroll_learned_params_save_filename, unroll_learned_params_concatenated)
    
    if (is_plotting):
        quat_dmp.plotDemosVsUnroll(set_qtraj_input, qdmp_unroll)
    
    return unroll_qtraj_concatenated, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI

if __name__ == "__main__":
    unroll_qtraj_concatenated, W, mean_A_learn, mean_tau, Ft, Fp, QgT, cX, cV, PSI = quat_dmp_multi_traj_training_test(time_reproduce_max=5.98734,#1.9976, 
                                                                                                                       tau_reproduce=5.98734,#1.9976, 
                                                                                                                       is_smoothing_training_traj_before_learning=False,#True, 
#                                                                                                                       percentage_padding=1.5, percentage_smoothing_points=3.0, 
#                                                                                                                       smoothing_mode=3, smoothing_cutoff_frequency=5.0, 
                                                                                                                       is_plotting=True)