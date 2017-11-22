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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from DataIO import *
from utilities import *

def prepareDemoDatasetLOAVicon(freq=300.0):
    trajs_extraction_version = 2
    dir_path = '../../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_vicon_static/'
    
    data_global_coord = {}
    data_global_coord["baseline"] = extractSetCartCoordTrajectories(dir_path + '/baseline/endeff_trajs/')
    obs_avoid_dominant_axis_annotation = np.loadtxt(dir_path + '/data_annotation_obs_avoid_dominant_axis.txt', dtype=int)
    obs_avoid_demo_consistency_annotation = np.loadtxt(dir_path + '/data_annotation_obs_avoid_consistency.txt')
    
    data_global_coord["dt"] = 1/freq
    data_global_coord["obs_avoid_var_descriptor"] = ['obs_markers_global_coord', 
                                                     'endeff_trajs',
                                                     'obs_avoid_dominant_axis',
                                                     'obs_avoid_demo_consistency']
    
    N_settings = countNumericSubdirs(dir_path)
    data_global_coord["obs_avoid"] = [[None] * N_settings for j in range(4)]
    
    for i in range(N_settings):
        setting_dir_path = dir_path + "/" + str(i + 1) + "/"
        data_global_coord["obs_avoid"][0][i] = np.loadtxt(setting_dir_path + data_global_coord["obs_avoid_var_descriptor"][0] + '.txt')
        data_global_coord["obs_avoid"][1][i] = extractSetCartCoordTrajectories(setting_dir_path + data_global_coord["obs_avoid_var_descriptor"][1] + '/')
        data_global_coord["obs_avoid"][2][i] = obs_avoid_dominant_axis_annotation[i]
        data_global_coord["obs_avoid"][3][i] = (obs_avoid_demo_consistency_annotation==(i+1)).any()
    
    return data_global_coord