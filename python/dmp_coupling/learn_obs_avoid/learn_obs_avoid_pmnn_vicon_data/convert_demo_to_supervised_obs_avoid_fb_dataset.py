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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../vicon/'))
from learnCartPrimitiveMultiOnLocalCoord import *
from DataIO import *
from utilities import *
from vicon_obs_avoid_utils import *


n_rfs = 25  # Number of basis functions used to represent the forcing term of DMP
c_order = 2 # DMP is using 2nd order canonical system

## Demo Dataset Preparation

data_global_coord = prepareDemoDatasetLOAVicon()

saveObj(data_global_coord, 'data_multi_demo_vicon_static_global_coord.pkl')

# end of Demo Dataset Preparation

## Baseline Primitive Learning

print('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...')
[cart_coord_dmp_baseline_params,
 cart_coord_dmp_baseline_unroll_global_traj,
 cart_coord_dmp_baseline_unroll_local_traj,
 cart_coord_dmp_baseline_unroll_forcing_term_traj] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord["baseline"], 
                                                                                         data_global_coord["dt"],
                                                                                         n_rfs, 
                                                                                         c_order)

# end of Baseline Primitive Learning