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
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_coupling/learn_obs_avoid/vicon/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from utilities import *
from vicon_obs_avoid_utils import *
from plot_sphere import *

def visualizeSetting(setting_no):
    vicon_marker_radius             = 15.0/2000.0   # in meter
    critical_position_marker_radius = 30/1000.0     # in meter
    
    D                               = 3
    
    baseline_demo_line_color_code = 'c'
    obs_avoid_demo_line_color_code = 'm'
    
    ## Data Creation/Loading
    
    data_dirpath_candidates = ['./learn_obs_avoid_pmnn_vicon_data/']
    filename = 'data_multi_demo_vicon_static_global_coord.pkl'
    data_filepath = ''
    for data_dirpath_candidate in data_dirpath_candidates:
        if (os.path.isfile(data_dirpath_candidate + filename)):
            data_filepath = data_dirpath_candidate + filename
            break
    
    if (data_filepath == ''):
        data_global_coord = prepareDemoDatasetLOAVicon()
        saveObj(data_global_coord, (data_dirpath_candidates[0] + filename))
    else:
        data_global_coord = loadObj(data_filepath)
    
    ## end of Data Creation/Loading
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if (setting_no > 0):    # indexing of the settings start at 1!
        for op in range(data_global_coord['obs_avoid'][0][setting_no-1].shape[0]):
            plot_sphere(ax, vicon_marker_radius, 
                        data_global_coord['obs_avoid'][0][setting_no-1][op,0], 
                        data_global_coord['obs_avoid'][0][setting_no-1][op,1], 
                        data_global_coord['obs_avoid'][0][setting_no-1][op,2], 'r')
        
        for i in range(len(data_global_coord['obs_avoid'][1][setting_no-1])):
            if (i == 0):
                obs_avoid_demo_label = 'obs avoid demos'
            else:
                obs_avoid_demo_label = ''
            ax.plot(data_global_coord['obs_avoid'][1][setting_no-1][i].X[0,:], 
                    data_global_coord['obs_avoid'][1][setting_no-1][i].X[1,:], 
                    data_global_coord['obs_avoid'][1][setting_no-1][i].X[2,:], 
                    color=obs_avoid_demo_line_color_code,
                    label=obs_avoid_demo_label)
    ax.legend()
    
    return None

if __name__ == "__main__":
    visualizeSetting(194)