#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:00:00 2019

@author: gsutanto
"""

import os
import sys
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
import utilities as py_util

catkin_ws_path = py_util.getCatkinWSPath()
sl_data_path = catkin_ws_path + "/install/bin/arm"

# initialization by removing all SL data files inside sl_data_path
initial_dfilepaths = [sl_data_path+'/'+f for f in os.listdir(sl_data_path) if re.match(r'd+\d{5,5}$', f)] # match the regex pattern of SL data files
for initial_dfilepath in initial_dfilepaths:
    os.remove(initial_dfilepath)