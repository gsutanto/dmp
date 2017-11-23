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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../cart_dmp/cart_coord_dmp'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from DMPTrajectory import *
from DMPState import *
from TauSystem import *
from DMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from CartesianCoordTransformer import *
from CartesianCoordDMP import *
from utilities import *

def learnCartPrimitiveMultiOnLocalCoord(self, cart_global_traj, train_data_dt,
                                        dmp_basis_funcs_size=25, canonical_order=2,
                                        ctraj_local_coordinate_frame_selection=GSUTANTO_LOCAL_COORD_FRAME,
                                        unroll_traj_length=-1,
                                        unroll_dt=None, 
                                        is_using_scaling=[False] * 3
                                        ):
    # is_using_scaling=[False] * 3 # default is NOT using scaling on DMPs
    if (unroll_dt == None):
        unroll_dt = train_data_dt
    
    D = 3
    N_demo = len(cart_global_traj)
    
    tau_sys = TauSystem(MIN_TAU)
    canonical_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    cart_dmp = CartesianCoordDMP(dmp_basis_funcs_size, canonical_sys_discr, ctraj_local_coordinate_frame_selection)
    cart_dmp.setScalingUsage(is_using_scaling)
    
    [tau_learn, critical_states_learn, 
     W, mean_A_learn, mean_tau, 
     Ft, Fp, 
     G, cX, cV, 
     PSI] = cart_dmp.learnGetDefaultUnrollParams(cart_global_traj, 1.0/train_data_dt)
    
    cart_coord_dmp_params_basic = {}
    # cart_coord_dmp_params_basic['dt'] = train_data_dt
    cart_coord_dmp_params_basic['dt'] = unroll_dt
    cart_coord_dmp_params_basic['model_size'] = dmp_basis_funcs_size
    cart_coord_dmp_params_basic['canonical_order'] = canonical_order
    cart_coord_dmp_params_basic['W'] = W
    
    unroll_cart_coord_params_basic = {}
    unroll_cart_coord_params_basic['mean_A_learn'] = mean_A_learn
    if (unroll_traj_length == -1):
        unroll_cart_coord_params_basic['mean_tau'] = mean_tau
    else:
        unroll_cart_coord_params_basic['mean_tau'] = 1.0 * unroll_dt * (unroll_traj_length - 1)
    unroll_cart_coord_params_basic['mean_start_global'] = cart_dmp.getMeanStartPosition()
    unroll_cart_coord_params_basic['mean_goal_global'] = cart_dmp.getMeanGoalPosition()
    unroll_cart_coord_params_basic['ctraj_local_coordinate_frame_selection'] = ctraj_local_coordinate_frame_selection
    
    [cart_coord_dmp_params, 
     cart_coord_dmp_unroll_fit_global_traj, 
     cart_coord_dmp_unroll_fit_local_traj, 
     Ffit]  = unrollCartPrimitiveOnLocalCoord( cart_coord_dmp_params_basic, 
                                               unroll_cart_coord_params_basic )
    
    return cart_coord_dmp_params, cart_coord_dmp_unroll_fit_global_traj, cart_coord_dmp_unroll_fit_local_traj, Ffit