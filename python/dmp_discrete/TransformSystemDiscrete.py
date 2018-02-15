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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_base/'))
from TransformationSystem import *
from FuncApproximatorDiscrete import *
from CanonicalSystemDiscrete import *

class TransformSystemDiscrete(TransformationSystem, object):
    'Class for transformation systems of discrete DMPs.'
    
    def __init__(self, dmp_num_dimensions_init, canonical_system_discrete, func_approximator_discrete, 
                 is_using_scaling_init, ts_alpha=25.0, ts_beta=25.0/4.0,
                 start_dmpstate_discrete=None, current_dmpstate_discrete=None, 
                 current_velocity_dmpstate_discrete=None, goal_system_discrete=None,
                 transform_couplers_list=[], name=""):
        super(TransformSystemDiscrete, self).__init__(dmp_num_dimensions_init,
                                                      canonical_system_discrete, func_approximator_discrete, 
                                                      start_dmpstate_discrete, current_dmpstate_discrete, 
                                                      current_velocity_dmpstate_discrete, goal_system_discrete,
                                                      transform_couplers_list, name)
        self.alpha = ts_alpha
        self.beta = ts_beta
        self.is_using_scaling = is_using_scaling_init
        self.A_learn = np.zeros((self.dmp_num_dimensions,1))
    
    def isValid(self):
        assert (super(TransformSystemDiscrete, self).isValid())
        assert (self.alpha > 0.0)
        assert (self.beta > 0.0)
        assert (len(self.is_using_scaling) == self.dmp_num_dimensions)
        assert (self.A_learn.shape[0] == self.dmp_num_dimensions)
        return True
    
    def start(self, start_state_init, goal_state_init):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        assert (start_state_init.isValid())
        assert (goal_state_init.isValid())
        assert (start_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        assert (goal_state_init.dmp_num_dimensions == self.dmp_num_dimensions)
        
        self.start_state = start_state_init
        self.setCurrentState(start_state_init)
        
        G_init = goal_state_init.getX()
        if (self.canonical_sys.order == 2):
            # Best option for Schaal's DMP Model using 2nd order canonical system:
            # Using goal evolution system initialized with the start position (state) as goal position (state),
            # which over time evolves toward a steady-state goal position (state).
            # The main purpose is to avoid discontinuous initial acceleration (see the math formula
            # of Schaal's DMP Model for more details).
            # Please also refer the initialization described in paper:
            # B. Nemec and A. Ude, “Action sequencing using dynamic movement
            # primitives,” Robotica, vol. 30, no. 05, pp. 837–846, 2012.
            x0 = start_state_init.getX()
            xd0 = start_state_init.getXd()
            xdd0 = start_state_init.getXdd()
            
            tau = self.tau_sys.getTauRelative()
            g0 = ((((tau*tau*xdd0) * 1.0 / self.alpha) + (tau*xd0)) * 1.0 / self.beta) + x0
            current_goal_state_init = DMPState(g0)
        elif (self.canonical_sys.order == 1):
            current_goal_state_init = goal_state_init
        
        self.goal_sys.start(current_goal_state_init, G_init)
        self.resetCouplingTerm()
        self.is_started = True
        
        assert (self.isValid()), "Post-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        return None
    
    def getNextState(self, dt):
        assert (self.is_started)
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        assert (dt > 0.0)
        
        tau = self.tau_sys.getTauRelative()
        forcing_term, basis_function_vector = self.func_approx.getForcingTerm()
        ct_acc, ct_vel = self.getCouplingTerm()
        for d in range(self.dmp_num_dimensions):
            if (self.is_using_coupling_term_at_dimension[d] == False):
                ct_acc[d,0] = 0.0
                ct_vel[d,0] = 0.0
        
        time = self.current_state.time
        x0 = self.start_state.X
        x = self.current_state.X
        xd = self.current_state.Xd
        xdd = self.current_state.Xdd
        
        v = self.current_velocity_state.X
        vd = self.current_velocity_state.Xd
        
        G = self.goal_sys.G
        g = self.goal_sys.current_goal_state.X
        
        x = x + (xd * dt)
        
        A = G - x0
        
        for d in range(self.dmp_num_dimensions):
            if (self.is_using_scaling[d]):
                if (np.fabs(self.A_learn[d,0]) < MIN_FABS_AMPLITUDE):
                    A[d,0] = 1.0
                else:
                    A[d,0] = A[d,0] * 1.0 / self.A_learn[d,0]
            else:
                A[d,0] = 1.0
        
        vd = ((self.alpha * ((self.beta * (g - x)) - v)) + (forcing_term * A) + ct_acc) * 1.0 / tau
        assert (np.isnan(vd).any() == False), "vd contains NaN!"
        
        xdd = vd * 1.0 / tau
        xd = (v + ct_vel) * 1.0 / tau
        
        v = v + (vd * dt)
        
        time = time + dt
        
        # self.current_state = DMPState(x, xd, xdd, time)
        # self.current_velocity_state = DMPState(v, vd, np.zeros((self.dmp_num_dimensions,1)), time)
        self.current_state.X = x
        self.current_state.Xd = xd
        self.current_state.Xdd = xdd
        self.current_state.time = time
        self.current_velocity_state.X = v
        self.current_velocity_state.Xd = vd
        self.current_velocity_state.time = time
        next_state = self.current_state
        
        return next_state, forcing_term, ct_acc, ct_vel, basis_function_vector
    
    def getGoalTrajAndCanonicalTrajAndTauAndALearnFromDemo(self, dmptrajectory_demo_local, robot_task_servo_rate, steady_state_goal_position_local=None):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        assert (dmptrajectory_demo_local.isValid())
        assert (dmptrajectory_demo_local.dmp_num_dimensions == self.dmp_num_dimensions)
        assert (robot_task_servo_rate > 0.0)
        
        traj_length = dmptrajectory_demo_local.getLength()
        start_dmpstate_demo_local = dmptrajectory_demo_local.getDMPStateAtIndex(0)
        if (steady_state_goal_position_local is None):
            goal_steady_dmpstate_demo_local = dmptrajectory_demo_local.getDMPStateAtIndex(traj_length-1)
        else:
            goal_steady_dmpstate_demo_local = DMPState(steady_state_goal_position_local)
        A_learn = goal_steady_dmpstate_demo_local.X - start_dmpstate_demo_local.X
        dt = (goal_steady_dmpstate_demo_local.time[0,0] - start_dmpstate_demo_local.time[0,0])/(traj_length - 1.0)
        if (dt <= 0.0):
            dt = 1.0/robot_task_servo_rate
        tau = goal_steady_dmpstate_demo_local.time[0,0] - start_dmpstate_demo_local.time[0,0]
        if (tau < MIN_TAU):
            tau = (1.0 * (traj_length - 1))/robot_task_servo_rate
        self.tau_sys.setTauBase(tau)
        self.canonical_sys.start()
        self.start(start_dmpstate_demo_local, goal_steady_dmpstate_demo_local)
        tau_relative = self.tau_sys.getTauRelative()
        X_list = [None] * traj_length
        V_list = [None] * traj_length
        G_list = [None] * traj_length
        for i in range(traj_length):
            x = self.canonical_sys.getCanonicalPosition()
            v = self.canonical_sys.getCanonicalVelocity()
            g = self.goal_sys.getCurrentGoalState().getX()
            X_list[i] = x
            V_list[i] = v
            G_list[i] = g
            
            self.canonical_sys.updateCanonicalState(dt)
            self.updateCurrentGoalState(dt)
        X = np.hstack(X_list).reshape((1,traj_length))
        V = np.hstack(V_list).reshape((1,traj_length))
        G = np.hstack(G_list)
        self.is_started = False
        self.canonical_sys.is_started = False
        self.goal_sys.is_started = False
        
        goal_position_trajectory = G
        canonical_position_trajectory = X # might be used later during learning
        canonical_velocity_trajectory = V # might be used later during learning
        return goal_position_trajectory, canonical_position_trajectory, canonical_velocity_trajectory, tau, tau_relative, A_learn
    
    def getTargetForcingTermTraj(self, dmptrajectory_demo_local, robot_task_servo_rate):
        G, cX, cV, tau, tau_relative, A_learn = self.getGoalTrajAndCanonicalTrajAndTauAndALearnFromDemo(dmptrajectory_demo_local, robot_task_servo_rate)
        
        T = dmptrajectory_demo_local.getX()
        Td = dmptrajectory_demo_local.getXd()
        Tdd = dmptrajectory_demo_local.getXdd()
        F_target = ((np.square(tau_relative) * Tdd) - (self.alpha * ((self.beta * (G - T)) - (tau_relative * Td))))
        
        return F_target, cX, cV, tau, tau_relative, A_learn, G
    
    def getTargetCouplingTermTraj(self, dmptrajectory_demo_local, robot_task_servo_rate, steady_state_goal_position_local):
        G, cX, cV, tau, tau_relative, A_learn = self.getGoalTrajAndCanonicalTrajAndTauAndALearnFromDemo(dmptrajectory_demo_local, robot_task_servo_rate, steady_state_goal_position_local)
        
        T = dmptrajectory_demo_local.getX()
        Td = dmptrajectory_demo_local.getXd()
        Tdd = dmptrajectory_demo_local.getXdd()
        F, PSI = self.func_approx.getForcingTermTraj(cX, cV)
        C_target = ((np.square(tau_relative) * Tdd) - (self.alpha * ((self.beta * (G - T)) - (tau_relative * Td))) - F)
        
        return C_target, F, PSI, cX, cV, tau, tau_relative, G
    
    def setScalingUsage(self, is_using_scaling_init):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        assert (len(is_using_scaling_init) == self.dmp_num_dimensions)
        self.is_using_scaling = is_using_scaling_init
        assert (self.isValid()), "Post-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        return None
    
    def getLearningAmplitude(self):
        return copy.copy(self.A_learn)
    
    def setLearningAmplitude(self, new_A_learn):
        assert (self.isValid()), "Pre-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        assert (new_A_learn.shape[0] == self.dmp_num_dimensions)
        self.A_learn = new_A_learn
        assert (self.isValid()), "Post-condition(s) checking is failed: this TransformSystemDiscrete is invalid!"
        return None