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
import time
import matplotlib.pyplot as plt
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

def dmp_1D_test(amd_clmc_dmp_home_dir_path="../../../", canonical_order=2, time_reproduce_max=0.0, time_goal_change=0.0, 
                new_goal_scalar=0.0, tau_reproduce=0.0, unroll_traj_save_dir_path="", unroll_traj_save_filename="", 
                is_smoothing_training_traj_before_learning=False, 
                percentage_padding=None, percentage_smoothing_points=None, 
                smoothing_mode=None, smoothing_cutoff_frequency=None, 
                is_plotting=False):
    task_servo_rate = 1000.0
    model_size = 25
    tau = MIN_TAU
    
    new_goal = np.zeros((1,1))
    new_goal[0,0] = new_goal_scalar
    
    assert ((canonical_order >= 1) and (canonical_order <= 2))
    assert (time_reproduce_max >= 0.0)
    assert (time_goal_change >= 0.0)
    assert (tau_reproduce >= 0.0)
    
    dt = 1.0/task_servo_rate
    tau_sys = TauSystem(dt, MIN_TAU)
    can_sys_discr = CanonicalSystemDiscrete(tau_sys, canonical_order)
    dmp_discrete_1D = DMPDiscrete1D(model_size, can_sys_discr)
    
    set_1Dtraj_input = dmp_discrete_1D.extractSetTrajectories(amd_clmc_dmp_home_dir_path + "/data/dmp_1D/sample_traj_1.txt")
    
    [critical_states_learn, 
     W, mean_A_learn, mean_tau, 
     Ft, Fp, G, cX, cV, PSI
     ] = dmp_discrete_1D.learnFromSetTrajectories(set_1Dtraj_input, task_servo_rate, 
                                                  is_smoothing_training_traj_before_learning=is_smoothing_training_traj_before_learning, 
                                                  percentage_padding=percentage_padding, 
                                                  percentage_smoothing_points=percentage_smoothing_points, 
                                                  smoothing_mode=smoothing_mode, 
                                                  smoothing_cutoff_frequency=smoothing_cutoff_frequency)
    
    ## Reproduce
    if (time_reproduce_max <= 0.0):
        time_reproduce_max = mean_tau
    if (new_goal[0,0] <= 0.0):
        new_goal = dmp_discrete_1D.getMeanGoalPosition()
    if (tau_reproduce <= 0.0):
        tau_reproduce = mean_tau
    tau = tau_reproduce
    dmp_unroll_init_parameters = DMPUnrollInitParams(critical_states_learn, tau)
    dmp_discrete_1D.startWithUnrollParams(dmp_unroll_init_parameters)
    
    unroll_traj_length = int(np.round(time_reproduce_max*task_servo_rate) + 1)
    unroll_traj = np.zeros((unroll_traj_length, 4))
    time_idx_goal_change = int(np.round(time_goal_change*task_servo_rate) + 1)
    #t0 = time.time()
    for i in xrange(unroll_traj_length):
        # testing goal change:
        if (i == time_idx_goal_change):
            dmp_discrete_1D.setNewSteadyStateGoalPosition(new_goal)
        
        [dmpstate, _, 
         _, _, 
         _] = dmp_discrete_1D.getNextState(dt, True)
        unroll_traj[i,0] = dmpstate.time[0,0]
        unroll_traj[i,1] = dmpstate.X[0,0]
        unroll_traj[i,2] = dmpstate.Xd[0,0]
        unroll_traj[i,3] = dmpstate.Xdd[0,0]
    #print ("elapsed time = " + str(time.time() - t0))
    
    if (os.path.isdir(unroll_traj_save_dir_path)):
        np.savetxt(unroll_traj_save_dir_path + "/" + unroll_traj_save_filename, unroll_traj)
    
    if (is_plotting):
        plt.close('all')
        
        dmp1D_unroll = dmp_discrete_1D.unroll(critical_states_learn, tau, time_reproduce_max, dt)
        
        dmp_discrete_1D.plotDemosVsUnroll(set_1Dtraj_input, dmp1D_unroll, fig_num_offset=10)
    
    return unroll_traj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI

if __name__ == "__main__":
    import pylab as pl
    
    pl.close('all')
    
    unroll_traj, W, mean_A_learn, mean_tau, Ft, Fp, G, cX, cV, PSI = dmp_1D_test(is_smoothing_training_traj_before_learning=False, 
                                                                                 percentage_padding=None, percentage_smoothing_points=None, 
                                                                                 smoothing_mode=None, smoothing_cutoff_frequency=None, 
                                                                                 is_plotting=True)
    
    task_servo_rate = 1000.0
    dt = 1.0/task_servo_rate
    tau = (cX.shape[1]-1) * dt
    time = np.arange(0.0, tau+dt, dt).reshape(1,cX.shape[1])
    
    pl.figure()
    pl.plot(time.T, cX.T, 'r')
    pl.xlabel('time')
    pl.ylabel('p')
    pl.title('Phase Variable p versus Time')
    
    pl.figure()
    pl.plot(time.T, cV.T, 'b')
    pl.xlabel('time')
    pl.ylabel('u')
    pl.title('Phase Velocity u versus Time')
    
    pl.figure()
    for i in range(0,PSI.shape[0],2):
        pl.plot(cX.T, PSI[i,:].T, label='psi_'+str(i))
    pl.xlabel('p')
    pl.ylabel('psi')
    pl.legend()
    pl.title('Phase RBF psi versus Phase Variable p')
    
    pl.figure()
    for i in range(0,PSI.shape[0],2):
        pl.plot(time.T, PSI[i,:].T, label='psi_'+str(i))
    pl.xlabel('time')
    pl.ylabel('psi')
    pl.legend()
    pl.title('Phase RBF psi versus Time')
    
    pl.show()