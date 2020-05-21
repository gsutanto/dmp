#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 08:00:00 2019

@author: gsutanto
"""

import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import rospy
from distutils import dir_util
from shutil import copyfile
from std_msgs.msg import Bool
from amd_clmc_ros_messages.msg import DMPRLTactileFeedbackRobotExecMode
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../reinforcement_learning/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from pi2 import Pi2
import utilities as py_util
import rl_tactile_fb_utils as rl_util

plt.close('all')

prims_tbi = [1, 2]
end_plot_iter = 2

rl_data = py_util.loadObj('./rl_data.pkl')
for prim_tbi in prims_tbi:
    rl_util.plotLearningCurve(rl_data=rl_data, prim_to_be_improved=prim_tbi, 
                              end_plot_iter=end_plot_iter, 
                              save_filepath='./learning_curve')