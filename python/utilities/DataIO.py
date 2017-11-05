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

def extractSetNDTrajectories(dir_or_file_path, N=-1):
    if (os.path.isdir(dir_or_file_path)):
        trajectory_files_list = glob.glob(dir_or_file_path+"/*.txt")
    else:
        assert (dir_or_file_path.endswith('.txt')), "Only *.txt files are supported."
        trajectory_files_list = [dir_or_file_path]
    trajectories_list = []
    for trajectory_file in trajectory_files_list:
        

dir_or_file_path = "../../data/cart_dmp/cart_coord_dmp/single_traj_training/"

a = (glob.glob(dir_or_file_path+"/*.txt"))
print a
for A in a:
    print A
    print (A.endswith('.txt'))