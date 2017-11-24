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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../cart_dmp/cart_coord_dmp/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from DMPState import *
from DMPTrajectory import *
from TransformSystemDiscrete import *
from CartesianCoordDMP import *
from CartesianCoordTransformer import *
from DataIO import *
from utilities import *

def constructObsAvoidViconFeatMat(demo_obs_avoid_traj_local,
                                  point_obstacles_cart_position_local,
                                  dt,
                                  cart_coord_dmp_baseline_params,
                                  cart_coord_dmp):
    
    cart_coord_dmp.setParams(cart_coord_dmp_baseline_params['W'], cart_coord_dmp_baseline_params['A_learn'])
    
    sub_X = []
    
    return sub_X