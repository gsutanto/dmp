#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:00:00 2019

@author: gsutanto
"""

import os
import sys
import time
import copy
import numpy as np
import numpy.linalg as npla
import scipy.io as sio
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Bool
from amd_clmc_ros_messages.msg import DMPRLTactileFeedbackRobotExecMode
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/clmcplot/'))
import utilities as py_util
import clmcplot_utils as clmcplot_util
import rl_tactile_fb_utils as rl_util

class RLTactileFeedback:
    def updateRobotReadyStatusCallback(self, robot_ready_notification_msg):
        print ("Robot is ready to accept command!")
        self.is_robot_ready = robot_ready_notification_msg.data
    
    def __init__(self, node_name="rl_tactile_feedback", loop_rate=100):
        self.is_robot_ready = False
        
        rospy.init_node(node_name)
        self.ros_rate = rospy.Rate(loop_rate)
        rospy.Subscriber("/cpp_client_to_py_master/robot_ready_notification", Bool, self.updateRobotReadyStatusCallback, tcp_nodelay=True)
        self.dmp_rl_tactile_fb_robot_exec_mode_msg_pub = rospy.Publisher("/py_master_to_cpp_client/dmp_rl_tactile_fb_robot_exec_mode", DMPRLTactileFeedbackRobotExecMode, queue_size=1, tcp_nodelay=True)
        
        plt.close('all')
        
        self.catkin_ws_path = py_util.getCatkinWSPath()
        self.sl_data_dirpath = self.catkin_ws_path + "/install/bin/arm"
        self.nominal_prims_params_dirpath       = "../../../../data/dmp_coupling/learn_tactile_feedback/scraping/learned_prims_params/"
        self.openloopequiv_prims_params_dirpath = "../../../../data/dmp_coupling/learn_tactile_feedback/scraping/reinforcement_learning/learned_prims_params/"
        self.initial_pmnn_params_dirpath        = "../../../../data/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/pmnn/cpp_models/"
        self.iter_pmnn_params_dirpath           = "../../../../data/dmp_coupling/learn_tactile_feedback/scraping/reinforcement_learning/neural_nets/pmnn/cpp_models/"
        self.outdata_dirpath = './'
        
        self.is_deleting_dfiles = False#True # TODO (remove this)
        self.is_smoothing_training_traj_before_learning = True
        self.is_unrolling_pi2_samples = True
        self.is_plotting = True#False
        
        self.N_total_sense_dimensionality = 45
        self.N_primitives = 3
        self.K_PI2_samples = 5#38#75 # K
        
        #self.prims_tbi = [1,2] # TODO (un-comment): 2nd and 3rd primitives are to-be-improved (tbi)
        self.prims_tbi = [1] # TODO (comment): for testing purpose we work on 2nd primitive only as the one to-be-improved (tbi)
        self.cart_dim_tbi_dict = {}
        self.cart_dim_tbi_dict["Quaternion"] = np.array([1]) # to-be-improved (tbi): Quaternion DMP, 2nd dimension
        self.cart_types_tbi_list = self.cart_dim_tbi_dict.keys()
        
        self.cost_threshold = [0.0, 18928850.8053, 11066375.797]
        
        # not sure if the nominal (original) primitives below is needed or not...:
        #nominal_cdmp_params = rl_util.loadPrimsParamsAsDictFromDirPath(nominal_prims_params_dirpath, N_primitives)
        
        if (self.is_deleting_dfiles): # TODO (remove this)
            # initialization by removing all SL data files inside sl_data_dirpath
            py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
        
        self.rl_data = {}
        
        self.count_pmnn_param_reuse = 0
        for self.prim_tbi in self.prims_tbi:
            self.rl_data[self.prim_tbi] = {}
            self.it = 0
            
            if (self.is_deleting_dfiles): # TODO (remove this)
                py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
            
            # TODO: Robot Execution: to evaluate initial cost on current closed-loop behavior 
            #                        (involving feedback model) by unrolling 3 times and averaging the cost
            
            # extract initial unrolling results: trajectories, sensor trace deviations, cost
            self.rl_data[self.prim_tbi][self.it] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                            N_primitives=self.N_primitives, 
                                                                                                            N_cost_components=self.N_total_sense_dimensionality)
            
            py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
            
            while (self.rl_data[self.prim_tbi][self.it]["mean_accum_cost"][self.prim_tbi] > self.cost_threshold[self.prim_tbi]): # while (J > threshold):
                plt.close('all')
                
                # convert current closed-loop behavior into an equivalent open-loop behavior on
                # the current (assumed-static) environment setting
                self.cdmp_trajs = rl_util.extractCartDMPTrajectoriesFromUnrollResults(self.rl_data[self.prim_tbi][self.it])
                [
                 self.rl_data[self.prim_tbi][self.it]["cdmp_params_all_dim_learned"], 
                 self.rl_data[self.prim_tbi][self.it]["cdmp_unroll_all_dim_learned"]
                ] = rl_util.learnCartDMPUnrollParams(self.cdmp_trajs, 
                                                     prim_to_be_learned="All", 
                                                     is_smoothing_training_traj_before_learning=self.is_smoothing_training_traj_before_learning, 
                                                     is_plotting=self.is_plotting)
                
                py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
                
                raw_input("Press [ENTER] to continue...")
                
                plt.close('all')
                
                if (self.is_deleting_dfiles): # TODO (remove this)
                    py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
                
                # TODO: Robot Execution: unroll DMP params of the (assumed) equivalent open-loop behavior, 
                #                        and measure cost J'
                
                # TODO: check (assert?) if J' is closely similar to J?
                
                # set to-be-perturbed DMP params as mean, and define the initial covariance matrix
                [self.param_mean, self.param_dim_list
                 ] = rl_util.extractParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["cdmp_params_all_dim_learned"], 
                                                       self.cart_dim_tbi_dict, self.cart_types_tbi_list, self.prim_tbi)
                self.param_init_std = rl_util.computeParamInitStdHeuristic(self.param_mean, 
                                                                           params_mean_extrema_to_init_std_factor=5.0)
                self.param_cov = np.diag(np.ones(len(self.param_mean)) * (self.param_init_std * self.param_init_std))
                
                self.param_samples = np.random.multivariate_normal(self.param_mean, self.param_cov, self.K_PI2_samples)
                
                self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"] = {}
                for self.k in range(self.K_PI2_samples):
                    self.param_sample = self.param_samples[self.k,:]
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k] = {}
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["cdmp_params_all_dim_learned"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it]["cdmp_params_all_dim_learned"])
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["cdmp_params_all_dim_learned"] = rl_util.updateParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["cdmp_params_all_dim_learned"], 
                                                                                                                                                         self.cart_dim_tbi_dict, 
                                                                                                                                                         self.cart_types_tbi_list, 
                                                                                                                                                         self.prim_tbi, 
                                                                                                                                                         self.param_sample, 
                                                                                                                                                         self.param_dim_list)
                    
                    while (not self.is_robot_ready):
                        print ("Waiting for the robot to be ready to accept command...")
                    
                    rl_util.savePrimsParamsFromDictAtDirPath(prims_params_dirpath=self.openloopequiv_prims_params_dirpath, 
                                                             cdmp_params=self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["cdmp_params_all_dim_learned"])
                    
                    self.dmp_rl_tactile_fb_robot_exec_mode_msg = DMPRLTactileFeedbackRobotExecMode()
                    self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_OPENLOOPEQUIV_DMP_ONLY
                    self.dmp_rl_tactile_fb_robot_exec_mode_msg.execute_behavior_until_prim_no = self.prim_tbi
                    self.dmp_rl_tactile_fb_robot_exec_mode_msg.description = "Load Perturbed Open-Loop-Equivalent Primitive Parameters, Sample # %d/%d, Execute until Prim. # %d" % ((self.k+1), self.K_PI2_samples, (self.prim_tbi+1))
                    
                    self.dmp_rl_tactile_fb_robot_exec_mode_msg_pub.publish(self.dmp_rl_tactile_fb_robot_exec_mode_msg)
                    
                    time.sleep(7) # sleep for several seconds, to make sure the parameters are loaded properly by the C++ code
                    
                    self.is_robot_ready = False
                
                if (self.is_unrolling_pi2_samples):
                    self.rl_data[self.prim_tbi][self.it]["PI2_unroll_samples"] = rl_util.unrollPI2ParamsSamples(pi2_params_samples=self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"], 
                                                                                                                prim_to_be_improved=self.prim_tbi, 
                                                                                                                cart_types_to_be_improved=self.cart_types_tbi_list, 
                                                                                                                pi2_unroll_mean=self.rl_data[self.prim_tbi][self.it]["cdmp_unroll_all_dim_learned"], 
                                                                                                                is_plotting=self.is_plotting)
                
                for self.k in range(self.K_PI2_samples):
                    if (self.is_deleting_dfiles): # TODO (remove this)
                        py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
                    # TODO: save these K perturbed DMP params (one at a time) as text files, to be loaded by C++ program and executed by the robot, to evaluate each of their costs.
                
                # TODO: summarize these K perturbed DMP params into mean_new and cov_new using PI2 update, based on each of their cost
                # TODO: save mean_new (which is a DMP params by itself) as text files, to be loaded by C++ program and executed by the robot, to evaluate its cost Nu times, to ensure the average cost is really lower than the original one
                
                py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
                
                raw_input("Press [ENTER] to continue...")

if __name__ == '__main__':
    rl_tactile_fb = RLTactileFeedback()