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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_1D/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from TauSystem import *
from DMPUnrollInitParams import *
from CanonicalSystemDiscrete import *
from DMPDiscrete1D import *
from utilities import *

def dmp_1D_test(canonical_order=2, time_reproduce_max=0.0, time_goal_change=0.0, 
                new_goal_scalar=0.0, tau_reproduce=0.0):
    task_servo_rate = 1000.0
    model_size = 25
    tau = MIN_TAU
    
    new_goal = np.zeros((1,1))
    new_goal[0,0] = new_goal_scalar
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    assert (time_reproduce_max >= 0.0)
    assert (time_goal_change >= 0.0)
    assert (tau_reproduce >= 0.0)
    
    tau_sys = TauSystem(MIN_TAU)
    can_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    dmp_discrete_1D = DMPDiscrete1D(model_size, can_sys_discr)
    [tau_learn, critical_states_learn, 
     W, mean_A_learn, mean_tau, 
     Ft, Fp, G, cX, cV, PSI] = dmp_discrete_1D.learnFromPath("../../../data/dmp_1D/sample_traj_1.txt", task_servo_rate)
    
    ## Reproduce
    if (time_reproduce_max <= 0.0):
        time_reproduce_max = tau_learn
    if (new_goal[0,0] <= 0.0):
        new_goal = dmp_discrete_1D.getMeanGoalPosition()
    if (tau_reproduce <= 0.0):
        tau_reproduce = tau_learn
    tau = tau_reproduce
    dmp_unroll_init_parameters = DMPUnrollInitParams(critical_states_learn, tau)
    dmp_discrete_1D.startWithUnrollParams(dmp_unroll_init_parameters)
    dt = 1.0/task_servo_rate
    list_time_and_dmpstate = []
    
    for i in range(int(np.round(time_reproduce_max*task_servo_rate) + 1)):
        if (i == int(np.round(time_goal_change*task_servo_rate) + 1)):
            dmp_discrete_1D.setNewSteadyStateGoalPosition(new_goal)
        
        [dmpstate, forcing_term, 
         ct_acc, ct_vel, 
         basis_function_vector] = dmp_discrete_1D.getNextState(dt, True)
        list_time_and_dmpstate.append(np.array([dmpstate.getTime()[0,0], dmpstate.getX()[0,0], dmpstate.getXd()[0,0], dmpstate.getXdd()[0,0]]))
    
    unroll_traj = np.vstack(list_time_and_dmpstate)
    return unroll_traj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI

if __name__ == "__main__":
    unroll_traj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI = dmp_1D_test()