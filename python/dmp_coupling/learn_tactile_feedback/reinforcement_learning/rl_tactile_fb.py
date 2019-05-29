#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:00:00 2019

@author: gsutanto
"""

import os
import sys
import re
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/clmcplot/'))
import utilities as py_util
import clmcplot_utils as clmcplot_util

catkin_ws_path = py_util.getCatkinWSPath()
sl_data_path = catkin_ws_path + "/install/bin/arm"

is_deleting_all_dfiles_at_init = False#True
N_total_sense_dimensionality = 45

if (is_deleting_all_dfiles_at_init):
    # initialization by removing all SL data files inside sl_data_path
    initial_dfilepaths = [sl_data_path+'/'+f for f in os.listdir(sl_data_path) if re.match(r'd+\d{5,5}$', f)] # match the regex pattern of SL data files
    for initial_dfilepath in initial_dfilepaths:
        os.remove(initial_dfilepath)

init_new_env_dfilepaths = [sl_data_path+'/'+f for f in os.listdir(sl_data_path) if re.match(r'd+\d{5,5}$', f)] # match the regex pattern of SL data files
for init_new_env_dfilepath in init_new_env_dfilepaths:
    print("Processing datafile %s..." % init_new_env_dfilepath)
    clmcfile = clmcplot_util.ClmcFile(init_new_env_dfilepath)
    prim_no = clmcfile.get_variables(["ul_curr_prim_no"])[0].T
    X_vector = np.vstack([clmcfile.get_variables(["X_vector_%02d" % i for i in range(N_total_sense_dimensionality)])]).T
    assert(X_vector.shape[0] == prim_no.shape[0])
    assert(X_vector.shape[1] == N_total_sense_dimensionality)