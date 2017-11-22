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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from CartesianCoordTransformer import *
from DMPTrajectory import *
from DMPState import *
from utilities import *

def learnCartPrimitiveMultiOnLocalCoord(self, cart_global_traj, train_data_dt,
                                        n_rfs, c_order,
                                        ctraj_local_coordinate_frame_selection=GSUTANTO_LOCAL_COORD_FRAME,
                                        unroll_traj_length=-1,
                                        unroll_dt=None, 
                                        is_using_scaling=[False] * 3
                                        ):
    # is_using_scaling=[False] * 3 # default is NOT using scaling on DMPs
    if (unroll_dt == None):
        unroll_dt = train_data_dt
    
    D = 3
    return None