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
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from utilities import *

def generate1stOrderTimeAffineDynSys(init_val, dt, tau, alpha):
    traj_length = int(round(tau/dt))
    
    traj = np.zeros(traj_length)
    
    x = init_val
    for t in range(traj_length):
        traj[t] = x
        x = (1 - ((dt/tau) * alpha)) * x
    
    return traj

if __name__ == "__main__":
    plt.close('all')
    base_tau = 0.5
    dt = 1/300.0
    alpha = 25.0/3.0 # similar to (1st order) canonical system's alpha
    
    plt_color_code = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    N_traj = len(plt_color_code)
    stretched_traj_length = int(round(base_tau/dt * (N_traj + 1)/2))
    
    trajs = [None] * N_traj
    data_output = [None] * N_traj
    data_input = [None] * N_traj
    
    ax_plt = [None] * 2
    plt_label = [None] * 2
    fig, (ax_plt[0], ax_plt[1]) = plt.subplots(2, sharex=True, sharey=True)
    
    for i in range(N_traj):
        tau = (i+1) * base_tau
        traj = generate1stOrderTimeAffineDynSys(1.0, dt, tau, alpha)
        trajs[i] = traj
        data_output[i] = traj[1:] - traj[:-1] # = C_{t} - C_{t-1}
        data_input[i] = ((dt/tau) * traj[:-1]) # (dt/tau) * C_{t-1}
        plt_label[0] = 'traj ' + str(i+1) + ', tau=' + str(tau) + 's'
        plt_label[1] = 'traj ' + str(i+1)
        stretched_traj = stretchTrajectory(traj, stretched_traj_length)
        ax_plt[0].plot(traj, 
                       color=plt_color_code[i],
                       label=plt_label[0])
        ax_plt[1].plot(stretched_traj, 
                       color=plt_color_code[i],
                       label=plt_label[1])
    ax_plt[0].set_title('Unstretched vs Stretched Trajectories of 1st Order Time-Affine Dynamical Systems, dt=(1/' + str(1.0/dt) + ')s')
    ax_plt[0].set_ylabel('Unstretched')
    ax_plt[1].set_ylabel('Stretched')
    ax_plt[1].set_xlabel('Time Index')
    for p in range(2):
        ax_plt[p].legend()
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
#            f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)