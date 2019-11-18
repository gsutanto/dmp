#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:00:00 2019

@author: gsutanto
"""

import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from distutils import dir_util
from shutil import copyfile
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../reinforcement_learning/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
import utilities as py_util
import pyplot_util as pypl_util
import rl_tactile_fb_utils as rl_util

generic_task_type = 'scraping'
specific_task_type = 'scraping_w_tool'
date = '20190902exp2'
additional_description = '_on_barrett_hand_368_trained_on_settings_4RLto6RL_rl_on_setting_8_reg_hidden_layer_100'
experiment_name = date + '_' + specific_task_type + '_correctable' + additional_description

user_home_dir_path = os.path.expanduser('~')
in_data_root_dir_path = user_home_dir_path + '/Desktop/dmp_robot_unroll_results/' + generic_task_type + '/' + experiment_name + '/robot/'

assert (os.path.isdir(in_data_root_dir_path)), '%s directory is not exist!' % in_data_root_dir_path

settings = ['p5', 'p6_25', 'p7_5', 'p8_75', 'p10']
unroll_types = ['bsln', 'cpld_before_rl', 'cpld_after_rl'] # 'bsln' = baseline; 'cpld_before_rl' = coupled before RL; 'cpld_after_rl' = coupled after RL

setting_to_roll_angle_mapping_dict = {}
setting_to_roll_angle_mapping_dict['p3_75'] = 3.75
setting_to_roll_angle_mapping_dict['p5'] = 5.0
setting_to_roll_angle_mapping_dict['p6_25'] = 6.25
setting_to_roll_angle_mapping_dict['p7_5'] = 7.5
setting_to_roll_angle_mapping_dict['p8_75'] = 8.75
setting_to_roll_angle_mapping_dict['p10'] = 10.0

unroll_type_to_eval_description_mapping_dict = {}
unroll_type_to_eval_description_mapping_dict['bsln'] = 'without feedback model'
unroll_type_to_eval_description_mapping_dict['cpld_before_rl'] = 'with feedback model before RL'
unroll_type_to_eval_description_mapping_dict['cpld_after_rl'] = 'with feedback model after RL'

roll_angles = np.array([setting_to_roll_angle_mapping_dict[setting] for setting in settings])
eval_descriptions = [unroll_type_to_eval_description_mapping_dict[unroll_type] for unroll_type in unroll_types]

N_prims = 3
N_total_sense_dimensionality = 45

prims_learned = [1,2]
X_lim_offset = 1.0

rl_robot_unroll_eval_data = {}
for setting in settings:
    rl_robot_unroll_eval_data[setting] = {}
    for unroll_type in unroll_types:
        rl_robot_unroll_eval_data[setting][unroll_type] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(in_data_root_dir_path+'/'+setting+'/'+unroll_type+'/', 
                                                                                                                   N_primitives=N_prims, 
                                                                                                                   N_cost_components=N_total_sense_dimensionality)

X_list = [roll_angles] * len(unroll_types)
X_lim = [np.min(roll_angles) - X_lim_offset, np.max(roll_angles) + X_lim_offset]
for prim in prims_learned:
    Y_list = list()
    E_list = list()
    for unroll_type in unroll_types:
        Y = list()
        E = list()
        for setting in settings:
            Y.append(rl_robot_unroll_eval_data[setting][unroll_type]['mean_accum_cost'][prim])
            E.append(rl_robot_unroll_eval_data[setting][unroll_type]['std_accum_cost'][prim])
        Y_list.append(Y)
        E_list.append(E)
    pypl_util.errorbar_2D(X_list=X_list, 
                          Y_list=Y_list, 
                          E_list=E_list, 
                          title='Performance at Primitive # %d : Cost for each considered Settings' % (prim+1), 
                          X_label='Environment Setting Roll-Angle (Degrees)', 
                          Y_label='Cost', 
                          fig_num=prim, 
                          label_list=eval_descriptions, 
                          color_style_list=[['r',None],['g',None],['b',None]], 
                          X_lim=X_lim, 
                          save_filepath='%s/plot_performance_before_vs_after_rl_of_primitive%d_%s' % (in_data_root_dir_path, prim+1, experiment_name)
                          )