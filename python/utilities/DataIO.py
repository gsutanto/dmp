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
from DMPTrajectory import *

def extractSetNDTrajectories(dir_or_file_path, N=-1):
    if (os.path.isdir(dir_or_file_path)):
        trajectory_files_list = glob.glob(dir_or_file_path+"/*.txt")
    else:
        assert (dir_or_file_path.endswith('.txt')), "Only *.txt files are supported."
        trajectory_files_list = [dir_or_file_path]
    trajectories_list = []
    for trajectory_file in trajectory_files_list:
        trajectory_file_content = np.loadtxt(trajectory_file)
        if (N == -1):
            N = (trajectory_file_content.shape[1] - 1)/3
        trajectory = DMPTrajectory(np.transpose(trajectory_file_content[:,1:N+1]), np.transpose(trajectory_file_content[:,N+1:(2*N)+1]), np.transpose(trajectory_file_content[:,(2*N)+1:(3*N)+1]), np.transpose(trajectory_file_content[:,0:1]))
        trajectories_list.append(trajectory)
    return trajectories_list