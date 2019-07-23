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
import matplotlib.pyplot as plt
import rospy
from distutils import dir_util
from std_msgs.msg import Bool
from amd_clmc_ros_messages.msg import DMPRLTactileFeedbackRobotExecMode
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../reinforcement_learning/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from pi2 import Pi2
import utilities as py_util
import rl_tactile_fb_utils as rl_util
import rl_tactile_fb_pmnn_supervised_training as rl_pmnn_tr

class RLTactileFeedback:
    def updateRobotReadyStatusCallback(self, robot_ready_notification_msg):
        self.is_robot_ready = robot_ready_notification_msg.data
    
    def executeBehaviorOnRobotNTimes(self, N_unroll, 
                                     exec_behavior_until_prim_no, 
                                     behavior_params=None, 
                                     feedback_model_params=None, 
                                     exec_mode="EXEC_OPENLOOPEQUIV_DMP_ONLY", 
                                     suffix_exec_description="", 
                                     periodic_wait_time_until_robot_is_ready_secs=0.0, 
                                     periodic_wait_time_until_robot_is_finished_processing_transmitted_cmd_secs=0.05, 
                                     wait_time_until_robot_is_finished_logging_into_datafile_secs=2.0
                                     ):
        # start by removing all SL data files inside sl_data_dirpath
        py_util.deleteAllCLMCDataFilesInDirectory(self.sl_data_dirpath)
        prev_clmc_dfilepaths_list = []
        n_unroll = 0
        
        while (n_unroll < N_unroll):
            is_waiting_robot_ready_msg_printed_out = False
            while (not self.is_robot_ready):
                if (not is_waiting_robot_ready_msg_printed_out):
                    print ("Waiting for the robot to be ready to accept command...")
                is_waiting_robot_ready_msg_printed_out = True
                if (periodic_wait_time_until_robot_is_ready_secs > 0.0):
                    time.sleep(periodic_wait_time_until_robot_is_ready_secs)
            
            if (exec_mode == "EXEC_OPENLOOPEQUIV_DMP_ONLY"):
                assert (behavior_params is not None)
                # save open-loop-equivalent primitive/behavior parameters into text files
                rl_util.savePrimsParamsFromDictAtDirPath(prims_params_dirpath=self.openloopequiv_prims_params_dirpath, 
                                                         cdmp_params=behavior_params)
            elif (exec_mode == "EXEC_NOMINAL_DMP_AND_ITERATION_PMNN"):
                assert (feedback_model_params is not None)
                # actually the (Iteration) PMNN parameters (C++ text files) have been updated when
                # the execution of trainPMNNWithAdditionalRLIterDatasetInitializedAtPath(...) is finished, 
                # so we do nothing here with the feedback_model_params ...
            
            # command the C++ side to load the text files containing the saved parameters and execute it on the robot
            self.dmp_rl_tactile_fb_robot_exec_mode_msg = DMPRLTactileFeedbackRobotExecMode()
            self.dmp_rl_tactile_fb_robot_exec_mode_msg.execute_behavior_until_prim_no = exec_behavior_until_prim_no
            if (exec_mode == "EXEC_NOMINAL_DMP_ONLY"):
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_NOMINAL_DMP_ONLY
                exec_description = "Open-Loop Behavior"
            elif (exec_mode == "EXEC_NOMINAL_DMP_AND_INITIAL_PMNN"):
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_NOMINAL_DMP_AND_INITIAL_PMNN
                exec_description = "Initial Closed-Loop Behavior"
            elif (exec_mode == "EXEC_OPENLOOPEQUIV_DMP_ONLY"):
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_OPENLOOPEQUIV_DMP_ONLY
                exec_description = "Open-Loop-Equivalent Behavior"
            elif (exec_mode == "EXEC_NOMINAL_DMP_AND_ITERATION_PMNN"):
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.rl_tactile_fb_robot_exec_mode = self.dmp_rl_tactile_fb_robot_exec_mode_msg.EXEC_NOMINAL_DMP_AND_ITERATION_PMNN
                exec_description = "Current Iteration Closed-Loop Behavior"
            else:
                assert (False), "exec_mode==%s is un-defined!" % exec_mode
            self.dmp_rl_tactile_fb_robot_exec_mode_msg.description = "Evaluating %s Trial # %d/%d, Execute until Prim. # %d" % (exec_description+suffix_exec_description, n_unroll+1, N_unroll, exec_behavior_until_prim_no+1)
            
            print (self.dmp_rl_tactile_fb_robot_exec_mode_msg.description)
            
            while (self.is_robot_ready):
                self.dmp_rl_tactile_fb_robot_exec_mode_msg.header.stamp = rospy.Time.now()
                self.dmp_rl_tactile_fb_robot_exec_mode_msg_pub.publish(self.dmp_rl_tactile_fb_robot_exec_mode_msg)
                print ("Waiting for the robot to finish processing transmitted command...")
                if (periodic_wait_time_until_robot_is_finished_processing_transmitted_cmd_secs > 0.0):
                    time.sleep(periodic_wait_time_until_robot_is_finished_processing_transmitted_cmd_secs)
            
            curr_clmc_dfilepaths_list = py_util.waitUntilTotalCLMCDataFilesReaches(self.sl_data_dirpath, n_unroll+1)
            new_clmc_dfilepath = list(set(curr_clmc_dfilepaths_list) - set(prev_clmc_dfilepaths_list))
            assert (len(new_clmc_dfilepath) == 1)
            new_clmc_dfilepath = new_clmc_dfilepath[0]
            
            time.sleep(wait_time_until_robot_is_finished_logging_into_datafile_secs)
            
            if (rl_util.checkUnrollResultCLMCDataFileValidity(new_clmc_dfilepath) == True):
                n_unroll += 1 # only advance counter if the latest-obtained CLMC datafile is valid
                prev_clmc_dfilepaths_list = copy.deepcopy(curr_clmc_dfilepaths_list)
            else:
                print ("The latest-obtained unroll result CLMC datafile %s is invalid!!! Deleting it and repeating the unroll..." % new_clmc_dfilepath)
                os.remove(new_clmc_dfilepath)
        
        assert (len(prev_clmc_dfilepaths_list) == N_unroll)
        
        return None
    
    def __init__(self, node_name="rl_tactile_feedback", loop_rate=100, 
                 is_unrolling_pi2_samples=True, 
                 is_plotting=True, 
                 starting_prim_tbi=-1, 
                 starting_rl_iter=-1):
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
        
        self.is_pipeline_executed_only_up_to_pi2 = False
        self.is_always_converting_new_ole_into_new_cl_every_rl_iters = True
        self.is_smoothing_training_traj_before_learning = True
        self.is_unrolling_pi2_samples = is_unrolling_pi2_samples
        self.is_plotting = is_plotting
        self.is_plotting_pi2_sample_before_robot_exec = True
        self.is_pausing = True#False
        
        self.N_total_sense_dimensionality = 45
        self.N_primitives = 3
        self.K_PI2_samples = 38#75 # K
        self.N_cost_evaluation_cl_behavior = 8
        self.N_cost_evaluation_ole_behavior_initial = 3
        self.N_cost_evaluation_ole_behavior_per_PI2_sample = 1
        self.N_cost_evaluation_ole_behavior_PI2_updated = 8
        
        self.cart_dim_tbi_dict = {}
        self.cart_dim_tbi_dict["Quaternion"] = np.array([1]) # to-be-improved (tbi): Quaternion DMP, 2nd dimension
        self.cart_types_tbi_list = self.cart_dim_tbi_dict.keys()
        
        self.cart_dim_tbi_supervision_threshold_dict = {}
        self.cart_dim_tbi_supervision_threshold_dict["Quaternion"] = {}
        self.cart_dim_tbi_supervision_threshold_dict["Quaternion"]["omegad"] = 5.0
        
#        self.cost_threshold = [0.0, 18928850.8053, 11066375.797] # X_vector squared norm as cost
        self.cost_threshold = [0.0, 0.05, 0.05] # rot_diff_err_b squared norm as cost
        
        self.pi2_opt = Pi2(kl_threshold = 1.0, covariance_damping = 2.0, 
                           is_computing_eta_per_timestep = True)
        
        self.prims_to_be_learned = [1,2]
        
        starting_prims_tbi = [1,2] # 2nd and 3rd primitives are to-be-improved (tbi)
        
        py_util.createDirIfNotExist(self.iter_pmnn_params_dirpath)
        dir_util.copy_tree(src=self.initial_pmnn_params_dirpath, dst=self.iter_pmnn_params_dirpath)
        
        self.nominal_cdmp_params = rl_util.loadPrimsParamsAsDictFromDirPath(self.nominal_prims_params_dirpath, self.N_primitives)
        
        if (not self.is_pipeline_executed_only_up_to_pi2):
            self.rl_tactile_fb_pmnn_supervised_training = rl_pmnn_tr.RLTactileFbPMNNSupervisedTraining()
        
        assert (len(starting_prims_tbi) <= self.N_primitives)
        assert ((np.array(starting_prims_tbi) >= 0).all() and (np.array(starting_prims_tbi) < self.N_primitives).all())
        assert (sorted(starting_prims_tbi) == starting_prims_tbi)
        assert (list(set(starting_prims_tbi)) == starting_prims_tbi)
        self.starting_prims_tbi_idx = 0
        if (starting_prim_tbi >= 0):
            assert (starting_prim_tbi < self.N_primitives)
            assert (starting_prim_tbi in starting_prims_tbi)
            self.starting_prims_tbi_idx = np.where(np.array(starting_prims_tbi) == starting_prim_tbi)[0][0]
        self.prims_tbi = starting_prims_tbi[self.starting_prims_tbi_idx:]
        
        print ("**********************************************************")
        print ("**********************************************************")
        print ("** Either execute the initial closed-loop (CL) behavior **")
        print ("** on the robot, or start from a particular             **")
        print ("** RL iteration by loading from a data file             **")
        print ("**********************************************************")
        print ("**********************************************************")
        if ((self.starting_prims_tbi_idx > 0) or (starting_rl_iter > 0)):
            self.rl_data = py_util.loadObj(self.outdata_dirpath+'rl_data.pkl')
        else:
            self.rl_data = {}
            
            self.executeBehaviorOnRobotNTimes(N_unroll=self.N_cost_evaluation_cl_behavior, 
                                              exec_behavior_until_prim_no=self.N_primitives - 1, 
                                              behavior_params=None, 
                                              feedback_model_params=None, 
                                              exec_mode="EXEC_NOMINAL_DMP_AND_INITIAL_PMNN")
        
        self.count_pmnn_param_reuse = 0
        for self.prim_tbi in self.prims_tbi:
            print ("**********************************************************")
            print ("**  Improving the feedback model of primitive # %d/%d...  **" % (self.prim_tbi+1, self.N_primitives))
            print ("**********************************************************")
            self.is_continuing_rl_iters = True
            
            if ((starting_rl_iter <= 0) or (self.prim_tbi != self.prims_tbi[0])):
                self.rl_data[self.prim_tbi] = {}
                
                self.it = 0
                
                self.rl_data[self.prim_tbi][self.it] = {}
                
                # extract initial unrolling results: trajectories, sensor trace deviations, cost
                self.rl_data[self.prim_tbi][self.it]["cl_cdmp_evals"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                 N_primitives=self.N_primitives, 
                                                                                                                                 N_cost_components=self.N_total_sense_dimensionality)
                
                self.rl_data[self.prim_tbi][self.it]["unroll_results"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it]["cl_cdmp_evals"])
                
                py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
            else:
                self.it = starting_rl_iter
            
#            while (self.rl_data[self.prim_tbi][self.it]["unroll_results"]["mean_accum_cost"][self.prim_tbi] > self.cost_threshold[self.prim_tbi]): # while (J > threshold):
            while (self.is_continuing_rl_iters):
                self.J = self.rl_data[self.prim_tbi][self.it]["unroll_results"]["mean_accum_cost"][self.prim_tbi]
                plt.close('all')
                
                print ("**********************************************************")
                print ("**********************************************************")
                print ("** Conversion of the current closed-loop (CL) behavior  **")
                print ("** into an open-loop-equivalent (OLE) behavior          **")
                print ("** in the current (assumed-static) environment setting  **")
                print ("** and evaluate the OLE behavior on the robot           **")
                print ("**********************************************************")
                print ("**********************************************************")
                [
                 self.rl_data[self.prim_tbi][self.it]["ole_cdmp_trajs"], 
                 self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params"], 
                 self.rl_data[self.prim_tbi][self.it]["ole_cdmp_unroll"]
                ] = rl_util.learnCartDMPUnrollParams(self.rl_data[self.prim_tbi][self.it]["unroll_results"], 
                                                     prims_to_be_learned=self.prims_to_be_learned, #"All", 
                                                     is_smoothing_training_traj_before_learning=self.is_smoothing_training_traj_before_learning, 
                                                     is_plotting=self.is_plotting, 
                                                     default_cdmp_params=self.nominal_cdmp_params)
                
                plt.close('all')
                
                self.executeBehaviorOnRobotNTimes(N_unroll=self.N_cost_evaluation_ole_behavior_initial, 
                                                  exec_behavior_until_prim_no=self.N_primitives - 1, 
                                                  behavior_params=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params"], 
                                                  feedback_model_params=None, 
                                                  exec_mode="EXEC_OPENLOOPEQUIV_DMP_ONLY", 
                                                  suffix_exec_description=" RL Iter. %d" % (self.it))
                
                # evaluate the cost of the open-loop-equivalent primitive
                self.rl_data[self.prim_tbi][self.it]["ole_cdmp_evals"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                  N_primitives=self.prim_tbi+1, 
                                                                                                                                  N_cost_components=self.N_total_sense_dimensionality)
                self.J_prime = self.rl_data[self.prim_tbi][self.it]["ole_cdmp_evals"]["mean_accum_cost"][self.prim_tbi]
                self.abs_diff_J_and_J_prime = np.fabs(self.J - self.J_prime)
                
                print ("prim_tbi # %d RL iter # %03d : J = %f ; J_prime = %f ; abs_diff_J_and_J_prime = %f" % (self.prim_tbi+1, self.it, self.J, self.J_prime, self.abs_diff_J_and_J_prime))
                print ("(should be roughly equivalent/close, which is indicative of equivalence between open-loop-equivalent (OLE) behavior and the closed-loop (CL) behavior...)")
                # TODO: check (assert?) if J' is closely similar to J?
                
                if (self.is_pausing):
                    raw_input("Press [ENTER] to continue...")
                
                # set to-be-perturbed DMP params as mean, and define the initial covariance matrix
                [self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], self.rl_data[self.prim_tbi][self.it]["PI2_param_dim_list"]
                 ] = rl_util.extractParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params"], 
                                                       self.cart_dim_tbi_dict, self.cart_types_tbi_list, self.prim_tbi)
                if (self.it == 0):
                    self.rl_data[self.prim_tbi][self.it]["PI2_param_init_std"] = 10.0 # rl_util.computeParamInitStdHeuristic(self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], params_mean_extrema_to_init_std_factor=5.0)
                    self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"] = np.diag(np.ones(len(self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"])) * (self.rl_data[self.prim_tbi][self.it]["PI2_param_init_std"] * self.rl_data[self.prim_tbi][self.it]["PI2_param_init_std"]))
                else:
                    self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"] = self.rl_data[self.prim_tbi][self.it-1]["PI2_param_new_cov"]
                
                self.param_samples = np.random.multivariate_normal(self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], 
                                                                   self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"], self.K_PI2_samples)
                
                print ("**********************************************************")
                print ("**********************************************************")
                print ("** PI2 sampling:                                        **")
                print ("** Sampling of the perturbed OLE behavior parameters    **")
                print ("** and evaluate each sample on the robot                **")
                print ("**********************************************************")
                print ("**********************************************************")
                self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"] = {}
                self.param_sample_cost_per_time_step_list = list()
                for self.k in range(self.K_PI2_samples):
                    self.param_sample = self.param_samples[self.k,:]
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k] = {}
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params"])
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params"] = rl_util.updateParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params"], 
                                                                                                                                             self.cart_dim_tbi_dict, 
                                                                                                                                             self.cart_types_tbi_list, 
                                                                                                                                             self.prim_tbi, 
                                                                                                                                             self.param_sample, 
                                                                                                                                             self.rl_data[self.prim_tbi][self.it]["PI2_param_dim_list"])
                
                if (self.is_unrolling_pi2_samples):
                    self.rl_data[self.prim_tbi][self.it]["PI2_unroll_samples"] = rl_util.unrollPI2ParamsSamples(pi2_params_samples=self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"], 
                                                                                                                prim_to_be_improved=self.prim_tbi, 
                                                                                                                cart_types_to_be_improved=self.cart_types_tbi_list, 
                                                                                                                pi2_unroll_mean=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_unroll"], 
                                                                                                                is_plotting=self.is_plotting)
                
                for self.k in range(self.K_PI2_samples):
                    if (self.is_plotting_pi2_sample_before_robot_exec or self.is_plotting):
                        rl_util.plotUnrollPI2ParamSampleVsParamMean(k=self.k, 
                                                                    prim_to_be_improved=self.prim_tbi, 
                                                                    cart_types_to_be_improved=self.cart_types_tbi_list, 
                                                                    pi2_unroll_samples=self.rl_data[self.prim_tbi][self.it]["PI2_unroll_samples"], 
                                                                    pi2_unroll_mean=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_unroll"])
                    
                    if (self.is_pausing and 
                        rl_util.checkUnrollPI2ParamSampleSupervisionRequirement(k=self.k, 
                                                                                cart_types_to_be_improved=self.cart_types_tbi_list, 
                                                                                cart_dim_tbi_supervision_threshold_dict=self.cart_dim_tbi_supervision_threshold_dict, 
                                                                                pi2_unroll_samples=self.rl_data[self.prim_tbi][self.it]["PI2_unroll_samples"])):
                        raw_input("Press [ENTER] to continue...")
                    
                    self.executeBehaviorOnRobotNTimes(N_unroll=self.N_cost_evaluation_ole_behavior_per_PI2_sample, 
                                                      exec_behavior_until_prim_no=self.prim_tbi, 
                                                      behavior_params=self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_params"], 
                                                      feedback_model_params=None, 
                                                      exec_mode="EXEC_OPENLOOPEQUIV_DMP_ONLY", 
                                                      suffix_exec_description=" RL Iter. %d PI2 Sample # %d/%d" % (self.it, self.k+1, self.K_PI2_samples))
                    
                    # evaluate the k-th sample's cost
                    self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_evals"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                                                    N_primitives=self.prim_tbi+1, 
                                                                                                                                                                    N_cost_components=self.N_total_sense_dimensionality)
                    self.param_sample_cost_per_time_step_list.append(self.rl_data[self.prim_tbi][self.it]["PI2_params_samples"][self.k]["ole_cdmp_evals"]["trajectory"][0]["cost_per_timestep"][self.prim_tbi])
                
                self.rl_data[self.prim_tbi][self.it]["PI2_param_sample_cost_per_time_step"] = np.vstack(self.param_sample_cost_per_time_step_list)
                
                print ("***********************************************************")
                print ("***********************************************************")
                print ("** PI2 update:                                           **")
                print ("** Updating the OLE behavior parameters to a better one  **")
                print ("** and evaluate it on the robot                          **")
                print ("***********************************************************")
                print ("***********************************************************")
                [self.rl_data[self.prim_tbi][self.it]["PI2_param_new_mean"], self.rl_data[self.prim_tbi][self.it]["PI2_param_new_cov"], _, _
                 ] = self.pi2_opt.update(self.param_samples, self.rl_data[self.prim_tbi][self.it]["PI2_param_sample_cost_per_time_step"], 
                                         self.rl_data[self.prim_tbi][self.it]["PI2_param_mean"], self.rl_data[self.prim_tbi][self.it]["PI2_param_cov"])
                
                self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_params"])
                self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"] = rl_util.updateParamsToBeImproved(self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"], 
                                                                                                               self.cart_dim_tbi_dict, 
                                                                                                               self.cart_types_tbi_list, 
                                                                                                               self.prim_tbi, 
                                                                                                               self.rl_data[self.prim_tbi][self.it]["PI2_param_new_mean"], 
                                                                                                               self.rl_data[self.prim_tbi][self.it]["PI2_param_dim_list"])
                
                if (self.is_pausing):
                    raw_input("About to execute PI2-updated open-loop-equivalent (OLE) behavior on the robot. Press [ENTER] to continue...")
                
                self.executeBehaviorOnRobotNTimes(N_unroll=self.N_cost_evaluation_ole_behavior_PI2_updated, 
                                                  exec_behavior_until_prim_no=self.N_primitives - 1, 
                                                  behavior_params=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_params"], 
                                                  feedback_model_params=None, 
                                                  exec_mode="EXEC_OPENLOOPEQUIV_DMP_ONLY", 
                                                  suffix_exec_description=" (after PI2 Update) RL Iter. %d" % (self.it))
                
                # evaluate the new sample mean's cost
                self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_evals"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                      N_primitives=self.N_primitives, 
                                                                                                                                      N_cost_components=self.N_total_sense_dimensionality)
                
                self.J_prime_new = self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_evals"]["mean_accum_cost"][self.prim_tbi]
                self.J_prime_new_minus_J_prime = self.J_prime_new - self.J_prime
                self.J_prime_new_minus_J = self.J_prime_new - self.J
                
                print ("prim_tbi # %d RL iter # %03d : J = %f ; J_prime = %f ; J_prime_new = %f" % (self.prim_tbi+1, self.it, self.J, self.J_prime, self.J_prime_new))
                print ("                             J_prime_new_minus_J_prime = %f" % self.J_prime_new_minus_J_prime)
                print ("                             J_prime_new_minus_J       = %f" % self.J_prime_new_minus_J)
                print ("(should all be negative (-), which is indicative of reinforcement learning progress...)")
                # TODO: check (assert?) if (J'new < J') and (J'new < J)?
                
                user_input = raw_input("Press n to stop current RL process (and continue to the RL process of the next primitive), or press anything else to continue current RL process...")
                if (user_input == 'n'):
                    self.is_continuing_rl_iters = False
                
                self.is_current_iter_converting_new_ole_into_new_cl = ((not self.is_pipeline_executed_only_up_to_pi2) and 
                                                                       (self.is_always_converting_new_ole_into_new_cl_every_rl_iters or 
                                                                        ((not self.is_always_converting_new_ole_into_new_cl_every_rl_iters) and 
                                                                         (not self.is_continuing_rl_iters))))
                
                if (self.is_current_iter_converting_new_ole_into_new_cl):
                    print ("**********************************************************")
                    print ("**********************************************************")
                    print ("** Conversion of the improved OLE behavior              **")
                    print ("** into an improved CL behavior                         **")
                    print ("** in the current (assumed-static) environment setting  **")
                    print ("** and evaluate the CL behavior on the robot            **")
                    print ("**********************************************************")
                    print ("**********************************************************")
                    [
                     self.rl_data[self.prim_tbi][self.it]["adapted_cdmp_trajs"], 
                     self.rl_data[self.prim_tbi][self.it]["additional_fb_dataset"]
                     ] = rl_util.extractAdditionalBehaviorFeedbackModelDataset(unroll_results=self.rl_data[self.prim_tbi][self.it]["ole_cdmp_new_evals"], 
                                                                               cdmp_params=self.nominal_cdmp_params, 
                                                                               is_smoothing_training_traj_before_learning=True)
                    
                    assert (self.it >= 0)
#                    if (self.it == 0):
#                        self.iterations_list = [0]
#                    else:
#                        self.iterations_list = [self.it-1, self.it]
                    self.iterations_list = [self.it]
                    
                    [
                     self.rl_data[self.prim_tbi][self.it]["updated_pmnn_params"], 
                     self.rl_data[self.prim_tbi][self.it]["updated_pmnn_eval_info"]
                     ] = self.rl_tactile_fb_pmnn_supervised_training.trainPMNNWithAdditionalRLIterDatasetInitializedAtPath(rl_data=self.rl_data, 
                                                                                                                           prim_tbi=self.prim_tbi, # prim-to-be-improved
                                                                                                                           iterations_list=self.iterations_list, 
                                                                                                                           initial_pmnn_params_dirpath=self.iter_pmnn_params_dirpath # should be cpp_models dirpath
                                                                                                                           )
                    
                    if (self.is_pausing):
                        raw_input("About to execute updated closed-loop (CL) behavior on the robot. Press [ENTER] to continue...")
                    
                    # robot execution of the new/updated (PMNN) behavior feedback model
                    self.executeBehaviorOnRobotNTimes(N_unroll=self.N_cost_evaluation_cl_behavior, 
                                                      exec_behavior_until_prim_no=self.N_primitives - 1, 
                                                      behavior_params=None, 
                                                      feedback_model_params=self.rl_data[self.prim_tbi][self.it]["updated_pmnn_params"], 
                                                      exec_mode="EXEC_NOMINAL_DMP_AND_ITERATION_PMNN")
                    
                    # extract unrolling results: trajectories, sensor trace deviations, cost
                    self.rl_data[self.prim_tbi][self.it]["cl_cdmp_new_evals"] = rl_util.extractUnrollResultsFromCLMCDataFilesInDirectory(self.sl_data_dirpath, 
                                                                                                                                         N_primitives=self.N_primitives, 
                                                                                                                                         N_cost_components=self.N_total_sense_dimensionality)
                    
                    self.J_new = self.rl_data[self.prim_tbi][self.it]["cl_cdmp_new_evals"]["mean_accum_cost"][self.prim_tbi]
                    self.J_new_minus_J_prime = self.J_new - self.J_prime
                    self.J_new_minus_J = self.J_new - self.J
                    self.abs_diff_J_new_and_J_prime_new = np.fabs(self.J_new - self.J_prime_new)
                    
                    print ("prim_tbi # %d RL iter # %03d : J_new = %f ; J_prime_new = %f ; abs_diff_J_new_and_J_prime_new = %f" % (self.prim_tbi+1, self.it, self.J_new, self.J_prime_new, self.abs_diff_J_new_and_J_prime_new))
                    print ("(should be roughly equivalent/close, which is indicative of equivalence between the improved open-loop-equivalent (OLE) behavior and the new closed-loop (CL) behavior...)")
                    
                    print ("prim_tbi # %d RL iter # %03d : J = %f ; J_new = %f ; J_prime = %f ; J_prime_new = %f" % (self.prim_tbi+1, self.it, self.J, self.J_new, self.J_prime, self.J_prime_new))
                    print ("                             J_prime_new_minus_J_prime = %f" % self.J_prime_new_minus_J_prime)
                    print ("                             J_prime_new_minus_J       = %f" % self.J_prime_new_minus_J)
                    print ("                             J_new_minus_J_prime       = %f" % self.J_new_minus_J_prime)
                    print ("                             J_new_minus_J             = %f" % self.J_new_minus_J)
                    print ("(should all be negative (-), which is indicative of reinforcement learning progress...)")
                    # TODO: check (assert?) if (Jnew < J)?
                
                rl_util.plotLearningCurve(rl_data=self.rl_data, prim_to_be_improved=self.prim_tbi, end_plot_iter=self.it, save_filepath=self.outdata_dirpath+'learning_curve')
                
                py_util.saveObj(self.rl_data, self.outdata_dirpath+'rl_data.pkl')
                
                if (self.is_pausing):
                    raw_input("Press [ENTER] to continue...")
                
                if (self.is_continuing_rl_iters):
                    self.it += 1
                    self.rl_data[self.prim_tbi][self.it] = {}
                    
                    if (self.is_current_iter_converting_new_ole_into_new_cl):
                        self.rl_data[self.prim_tbi][self.it]["unroll_results"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it-1]["cl_cdmp_new_evals"])
                    else:
                        # extract unrolling results: trajectories, sensor trace deviations, cost
                        self.rl_data[self.prim_tbi][self.it]["unroll_results"] = copy.deepcopy(self.rl_data[self.prim_tbi][self.it-1]["ole_cdmp_new_evals"])
                
                # TODO: In simulation: test 4 different combinations of self.is_pipeline_executed_only_up_to_pi2 and self.is_always_converting_new_ole_into_new_cl_every_rl_iters until switching to the next primitive; all must be working successfully as expected (Saturday, July 20, 2019)

if __name__ == '__main__':
    rl_tactile_fb = RLTactileFeedback(is_unrolling_pi2_samples=True, 
                                      is_plotting=False, 
                                      starting_prim_tbi=1, 
                                      starting_rl_iter=0)