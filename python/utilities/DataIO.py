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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from DMPTrajectory import *
from utilities import *

def extractSetNDTrajectories(dir_or_file_path, N=-1):
    if (os.path.isdir(dir_or_file_path)):
        trajectory_files_list = naturalSort(glob.glob(dir_or_file_path+"/*.txt"))
    else:
        assert (dir_or_file_path.endswith('.txt')), "Only *.txt files are supported."
        trajectory_files_list = [dir_or_file_path]
    trajectories_list_size = len(trajectory_files_list)
    trajectories_list = [None] * trajectories_list_size
    for i in range(trajectories_list_size):
        trajectory_file = trajectory_files_list[i]
        trajectory_file_content = np.loadtxt(trajectory_file)
        if (N == -1):
            N = (trajectory_file_content.shape[1] - 1)/3
        trajectory = DMPTrajectory(np.transpose(trajectory_file_content[:,1:N+1]), np.transpose(trajectory_file_content[:,N+1:(2*N)+1]), np.transpose(trajectory_file_content[:,(2*N)+1:(3*N)+1]), np.transpose(trajectory_file_content[:,0:1]))
        trajectories_list[i] = trajectory
    return trajectories_list