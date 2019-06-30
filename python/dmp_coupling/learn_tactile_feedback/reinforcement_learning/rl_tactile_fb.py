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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../reinforcement_learning/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/clmcplot/'))
from pi2 import Pi2
import utilities as py_util
import clmcplot_utils as clmcplot_util
import rl_tactile_fb_utils as rl_util

class RLTactileFeedback:
    def updateRobotReadyStatusCallback(self, robot_ready_notification_msg):
        if (self.is_executing_on_robot):
            self.is_robot_ready = robot_ready_notification_msg.data
    
    def __init__(self, node_name="rl_tactile_feedback", loop_rate=100, 
                 is_executing_on_robot=False, 
                 is_unrolling_pi2_samples=True, 
                 is_plotting=True):
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
        
        self.is_smoothing_training_traj_before_learning = True
        self.is_executing_on_robot = is_executing_on_robot
        self.is_unrolling_pi2_samples = is_unrolling_pi2_samples
        self.is_plotting = is_plotting
        
        self.N_total_sense_dimensionality = 45
        self.N_primitives = 3
        self.K_PI2_samples = 5#38#75 # K
        self.N_cost_evaluation_general = 5
        self.N_cost_evaluation_per_PI2_sample = 1
        
        #self.prims_tbi = [1,2] # TODO (un-comment): 2nd and 3rd primitives are to-be-improved (tbi)
        self.prims_tbi = [1] # TODO (comment): for testing purpose we work on 2nd primitive only as the one to-be-improved (tbi)
        self.cart_dim_tbi_dict = {}
        self.cart_dim_tbi_dict["Quaternion"] = np.array([1]) # to-be-improved (tbi): Quaternion DMP, 2nd dimension
        self.cart_types_tbi_list = self.cart_dim_tbi_dict.keys()
        
        self.cost_threshold = [0.0, 18928850.8053, 11066375.797]
        
        self.pi2_opt = Pi2(kl_threshold = 1.0, covariance_damping = 2.0, 
                           is_computing_eta_per_timestep = True)
        
        # not sure if the nominal (original) primitives below is needed or not...:
        #nominal_cdmp_params = rl_util.loadPrimsParamsAsDictFromDirPath(nominal_prims_params_dirpath, N_primitives)
        
        if (self.is_executing_on_robot):
            # initialization by removing all SL data files inside sl_data_dirpath
            py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
            
            for self.n_cost_evaluation_general in range(self.N_cost_evaluation_general):
                while (not self.is_robot_ready):
                    print ("Waiting for the robot to be ready to accept command...")
                
                print ("Evaluating Closed-Loop Behavior, Execute All Primitives")
                
                # command the C++ side to load the text files containing the parameters and execute it on the robot
                self.dmp_rl_tactile_fb_robot_exec_mode_msg = DMPRLTactileFeedbackRobotExecMode()
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_NOMINAL_DMP_AND_INITIAL_PMNN
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.execute_behavior_until_prim_no = self.N_primitives - 1
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.description = "Load Closed-Loop Behavior Parameters, Execute All Primitives"
                
                self.dmp_rl_tactile_fb_robot_exec_mode_msg_pub.publish(self.dmp_rl_tactile_fb_robot_exec_mode_msg)
                
                while (self.is_robot_ready):
                    print ("Waiting for the robot to finish processing transmitted command...")
                    time.sleep(1)
                
                py_util.waitUntilTotalCLMCDataFilesReaches(self.sl_data_dirpath, self.n_cost_evaluation_general+1)
        
        self.rl_data = {}
        
        self.count_pmnn_param_reuse = 0
        for self.prim_tbi in self.prims_tbi:
            self.rl_data[self.prim_tbi] = {}
            self.it = 0
            self.rl_data[self.prim_tbi][self.it] = {}
            
            # extract initial unrolling results: trajectories, sensor trace deviations, cost
            self.rl_data[self.prim_tbi][self.it]["unroll_results"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                              N_primitives=self.N_primitives, 
                                                                                                                              N_cost_components=self.N_total_sense_dimensionality)
            
            py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
            
            while (self.rl_data[self.prim_tbi][self.it]["unroll_results"]["mean_accum_cost"][self.prim_tbi] > self.cost_threshold[self.prim_tbi]): # while (J > threshold):
                self.J = self.rl_data[self.prim_tbi][self.it]["unroll_results"]["mean_accum_cost"][self.prim_tbi]
                plt.close('all')
                
                # convert current closed-loop behavior into an open-loop-equivalent (ole) behavior on
                # the current (assumed-static) environment setting
                self.cdmp_trajs = rl_util.extractCartDMPTrajectoriesFromUnrollResults(self.rl_data[self.prim_tbi][self.it]["unroll_results"])
                [
                 self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params_all_dim_learned"], 
                 self.rl_data[self.prim_tbi][self.it]["ole_cdmp_unroll_all_dim_learned"]
                ] = rl_util.learnCartDMPUnrollParams(self.cdmp_trajs, 
                                                     prim_to_be_learned="All", 
                                                     is_smoothing_training_traj_before_learning=self.is_smoothing_training_traj_before_learning, 
                                                     is_plotting=self.is_plotting)
                
                py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
                
                raw_input("Press [ENTER] to continue...")
                
                plt.close('all')
                
                if (self.is_executing_on_robot):
                    py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
                    
                    for self.n_cost_evaluation_general in range(self.N_cost_evaluation_general):
                        while (not self.is_robot_ready):
                            print ("Waiting for the robot to be ready to accept command...")
                        
                        print ("Evaluating Open-Loop-Equivalent Primitive, Execute until Prim. # %d" % (self.prim_tbi+1))
                        
                        # save open-loop-equivalent primitive parameters into text files
                        rl_util.savePrimsParamsFromDictAtDirPath(prims_params_dirpath=self.openloopequiv_prims_params_dirpath, 
                                                                 cdmp_params=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params_all_dim_learned"])
                        
                        # command the C++ side to load the text files containing the saved parameters and execute it on the robot
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg = DMPRLTactileFeedbackRobotExecMode()
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_OPENLOOPEQUIV_DMP_ONLY
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg.execute_behavior_until_prim_no = self.prim_tbi
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg.description = "Load Open-Loop-Equivalent Primitive Parameters, Execute until Prim. # %d" % (self.prim_tbi+1)
                        
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg_pub.publish(self.dmp_rl_tactile_fb_robot_exec_mode_msg)
                        
                        while (self.is_robot_ready):
                            print ("Waiting for the robot to finish processing transmitted command...")
                            time.sleep(1)
                        
                        py_util.waitUntilTotalCLMCDataFilesReaches(self.sl_data_dirpath, self.n_cost_evaluation_general+1)
                
                # evaluate the cost of the open-loop-equivalent primitive
                self.rl_data[self.prim_tbi][self.it]["ole_cdmp_evals_all_dim_learned"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                                  N_primitives=self.prim_tbi+1, 
                                                                                                                                                  N_cost_components=self.N_total_sense_dimensionality)
                self.J_prime = self.rl_data[self.prim_tbi][self.it]["ole_cdmp_evals_all_dim_learned"]["mean_accum_cost"][self.prim_tbi]
                self.abs_diff_J_and_J_prime = np.fabs(self.J - self.J_prime)
                
                print ("prim_tbi # %d : J = %f ; J_prime = %f ; abs_diff_J_and_J_prime = %f" % (self.prim_tbi+1, self.J, self.J_prime, self.abs_diff_J_and_J_prime))
                raw_input("Press [ENTER] to continue...")
                # TODO: check (assert?) if J' is closely similar to J?
                
                # set to-be-perturbed DMP params as mean, and define the initial covariance matrix
                [self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], self.rl_data[self.prim_tbi][self.it]["PI2_param_dim_list"]
                 ] = rl_util.extractParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params_all_dim_learned"], 
                                                       self.cart_dim_tbi_dict, self.cart_types_tbi_list, self.prim_tbi)
                if (self.it == 0):
                    self.rl_data[self.prim_tbi][self.it]["PI2_param_init_std"] = rl_util.computeParamInitStdHeuristic(self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], 
                                                                                                                      params_mean_extrema_to_init_std_factor=5.0)
                    self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"] = np.diag(np.ones(len(self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"])) * (self.rl_data[self.prim_tbi][self.it]["PI2_param_init_std"] * self.rl_data[self.prim_tbi][self.it]["PI2_param_init_std"]))
                else:
                    self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"] = self.rl_data[self.prim_tbi][self.it-1]["PI2_param_new_cov"]
                
                self.param_samples = np.random.multivariate_normal(self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], 
                                                                   self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"], self.K_PI2_samples)
                
                self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"] = {}
                self.param_sample_cost_per_time_step_list = list()
                for self.k in range(self.K_PI2_samples):
                    self.param_sample = self.param_samples[self.k,:]
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k] = {}
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params_all_dim_learned"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params_all_dim_learned"])
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params_all_dim_learned"] = rl_util.updateParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params_all_dim_learned"], 
                                                                                                                                                             self.cart_dim_tbi_dict, 
                                                                                                                                                             self.cart_types_tbi_list, 
                                                                                                                                                             self.prim_tbi, 
                                                                                                                                                             self.param_sample, 
                                                                                                                                                             self.rl_data[self.prim_tbi][self.it]["PI2_param_dim_list"])
                    
                    if (self.is_executing_on_robot):
                        py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
                        
                        for self.n_cost_evaluation_per_PI2_sample in range(self.N_cost_evaluation_per_PI2_sample):
                            while (not self.is_robot_ready):
                                print ("Waiting for the robot to be ready to accept command...")
                            
                            print ("Evaluating PI2 Perturbed Open-Loop-Equivalent Primitive Sample # %d/%d, Execute until Prim. # %d" % ((self.k+1), self.K_PI2_samples, (self.prim_tbi+1)))
                            
                            # save sampled perturbed open-loop-equivalent primitive parameters into text files
                            rl_util.savePrimsParamsFromDictAtDirPath(prims_params_dirpath=self.openloopequiv_prims_params_dirpath, 
                                                                     cdmp_params=self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params_all_dim_learned"])
                            
                            # command the C++ side to load the text files containing the saved parameters and execute it on the robot
                            self.dmp_rl_tactile_fb_robot_exec_mode_msg = DMPRLTactileFeedbackRobotExecMode()
                            self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_OPENLOOPEQUIV_DMP_ONLY
                            self.dmp_rl_tactile_fb_robot_exec_mode_msg.execute_behavior_until_prim_no = self.prim_tbi
                            self.dmp_rl_tactile_fb_robot_exec_mode_msg.description = "Load Perturbed Open-Loop-Equivalent Primitive Parameters, Sample # %d/%d, Execute until Prim. # %d" % ((self.k+1), self.K_PI2_samples, (self.prim_tbi+1))
                            
                            self.dmp_rl_tactile_fb_robot_exec_mode_msg_pub.publish(self.dmp_rl_tactile_fb_robot_exec_mode_msg)
                            
                            while (self.is_robot_ready):
                                print ("Waiting for the robot to finish processing transmitted command...")
                                time.sleep(1)
                            
                            py_util.waitUntilTotalCLMCDataFilesReaches(self.sl_data_dirpath, self.n_cost_evaluation_per_PI2_sample+1)
                    
                    # evaluate the k-th sample's cost
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_evals_all_dim_learned"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                                                                    N_primitives=self.prim_tbi+1, 
                                                                                                                                                                                    N_cost_components=self.N_total_sense_dimensionality)
                    self.param_sample_cost_per_time_step_list.append(self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_evals_all_dim_learned"]["trajectory"][0]["cost_per_timestep"][self.prim_tbi])
                
                self.rl_data[self.prim_tbi][self.it]["PI2_param_sample_cost_per_time_step"] = np.vstack(self.param_sample_cost_per_time_step_list)
                
                [self.rl_data[self.prim_tbi][self.it]["PI2_param_new_mean"], self.rl_data[self.prim_tbi][self.it]["PI2_param_new_cov"], _, _
                 ] = self.pi2_opt.update(self.param_samples, self.rl_data[self.prim_tbi][self.it]["PI2_param_sample_cost_per_time_step"], 
                                         self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"])
                
                if (self.is_unrolling_pi2_samples):
                    self.rl_data[self.prim_tbi][self.it]["PI2_unroll_samples"] = rl_util.unrollPI2ParamsSamples(pi2_params_samples=self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"], 
                                                                                                                prim_to_be_improved=self.prim_tbi, 
                                                                                                                cart_types_to_be_improved=self.cart_types_tbi_list, 
                                                                                                                pi2_unroll_mean=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_unroll_all_dim_learned"], 
                                                                                                                is_plotting=self.is_plotting)
                
                self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params_all_dim_learned"])
                self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"] = rl_util.updateParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"], 
                                                                                                               self.cart_dim_tbi_dict, 
                                                                                                               self.cart_types_tbi_list, 
                                                                                                               self.prim_tbi, 
                                                                                                               self.rl_data[self.prim_tbi][self.it]["PI2_param_new_mean"], 
                                                                                                               self.rl_data[self.prim_tbi][self.it]["PI2_param_dim_list"])
                
                # save mean_new (which is a DMP params by itself) as text files, to be loaded by C++ program and executed by the robot, to evaluate its cost Nu times, to ensure the average cost is really lower than the original one
                if (self.is_executing_on_robot):
                    py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
                    
                    for self.n_cost_evaluation_general in range(self.N_cost_evaluation_general):
                        while (not self.is_robot_ready):
                            print ("Waiting for the robot to be ready to accept command...")
                        
                        print ("Evaluating Open-Loop-Equivalent Primitive New Sample Mean, Execute until Prim. # %d" % (self.prim_tbi+1))
                        
                        # save open-loop-equivalent primitive's new sample mean parameters into text files
                        rl_util.savePrimsParamsFromDictAtDirPath(prims_params_dirpath=self.openloopequiv_prims_params_dirpath, 
                                                                 cdmp_params=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"])
                        
                        # command the C++ side to load the text files containing the saved parameters and execute it on the robot
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg = DMPRLTactileFeedbackRobotExecMode()
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_OPENLOOPEQUIV_DMP_ONLY
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg.execute_behavior_until_prim_no = self.prim_tbi
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg.description = "Load Open-Loop-Equivalent Primitive New Sample Mean Parameters, Execute until Prim. # %d" % (self.prim_tbi+1)
                        
                        self.dmp_rl_tactile_fb_robot_exec_mode_msg_pub.publish(self.dmp_rl_tactile_fb_robot_exec_mode_msg)
                        
                        while (self.is_robot_ready):
                            print ("Waiting for the robot to finish processing transmitted command...")
                            time.sleep(1)
                        
                        py_util.waitUntilTotalCLMCDataFilesReaches(self.sl_data_dirpath, self.n_cost_evaluation_general+1)
                
                py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
                
                raw_input("Press [ENTER] to continue...")
                
                self.it += 1
                self.rl_data[self.prim_tbi][self.it] = {}
                
                # extract unrolling results: trajectories, sensor trace deviations, cost
                self.rl_data[self.prim_tbi][self.it]["unroll_results"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                  N_primitives=self.N_primitives, 
                                                                                                                                  N_cost_components=self.N_total_sense_dimensionality)
                
                py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')

if __name__ == '__main__':
    rl_tactile_fb = RLTactileFeedback(is_executing_on_robot=True, 
                                      is_unrolling_pi2_samples=False, 
                                      is_plotting=False)