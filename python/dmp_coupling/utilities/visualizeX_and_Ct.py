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
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from utilities import *

def visualizeX_and_Ct(setting_numbers, data_filepath, is_time_stretched=True, is_plotting_Ct=True, is_plotting_X=False, stretched_traj_length=600):
    D = 3
    
    baseline_demo_line_color_code = 'c'
    obs_avoid_demo_line_color_code = 'm'
    
    ## Data Creation/Loading
    
    assert (os.path.isfile(data_filepath))
    dataset_Ct = loadObj(data_filepath)
    max_setting_no = len(dataset_Ct['sub_Ct_target'][0])
    
    if (is_time_stretched):
        is_time_stretched_label = ', time-stretched'
    else:
        is_time_stretched_label = ''
    
    for setting_no in setting_numbers:
        assert (setting_no > 0), 'Indexing of the settings start at 1!'
        assert (setting_no <= max_setting_no), 'Maximum Setting Number is ' + str(max_setting_no) + '!'
        if (is_plotting_Ct):
            ax_ct = [None] * D
            fig_ct, (ax_ct[0], ax_ct[1], ax_ct[2]) = plt.subplots(3, sharex=True, sharey=True)
            
            for i in range(len(dataset_Ct['sub_Ct_target'][0][setting_no-1])):
                if (i == 0):
                    obs_avoid_demo_label = 'demos'
                else:
                    obs_avoid_demo_label = ''
                for d in range(D):
                    if (is_time_stretched):
                        traj = stretchTrajectory(dataset_Ct['sub_Ct_target'][0][setting_no-1][i][d,:], stretched_traj_length)
                    else:
                        traj = dataset_Ct['sub_Ct_target'][0][setting_no-1][i][d,:]
                    ax_ct[d].plot(traj, 
                                  color=obs_avoid_demo_line_color_code,
                                  label=obs_avoid_demo_label)
            ax_ct[0].set_title('Coupling Term for Setting #' + str(setting_no) + is_time_stretched_label)
            ax_ct[0].set_ylabel('x')
            ax_ct[1].set_ylabel('y')
            ax_ct[2].set_ylabel('z')
            ax_ct[2].set_xlabel('Time Index')
            for d in range(D):
                ax_ct[d].legend()
            # Fine-tune figure; make subplots close to each other and hide x ticks for
            # all but bottom plot.
#            f.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in fig_ct.axes[:-1]], visible=False)
    
    return None

if __name__ == "__main__":
    plt.close('all')
    visualizeX_and_Ct([120, 194], '../learn_obs_avoid/learn_obs_avoid_pmnn_vicon_data/dataset_Ct_obs_avoid.pkl', True)