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
import scipy.io as sio

def prepareData(task_type, dataset_Ct, subset_settings_indices,
                considered_subset_outlier_ranked_demo_indices,
                generalization_subset_outlier_ranked_demo_indices,
                post_filename_stacked_data,
                out_data_dir=''):
    feature_type = 'raw'
    mode_stack_dataset = 1
    
    N_primitive = len(dataset_Ct["sub_Ct_target"])
    
    training_subset_outlier_ranked_demo_indices = list(set(considered_subset_outlier_ranked_demo_indices) - set(generalization_subset_outlier_ranked_demo_indices))
    if (generalization_subset_outlier_ranked_demo_indices == []):
        generalization_subset_outlier_ranked_demo_indices = [1] # CANNOT really be empty (for further Python processing)
    
    list_subset_outlier_ranked_demo_indices = [training_subset_outlier_ranked_demo_indices, 
                                               generalization_subset_outlier_ranked_demo_indices]
    list_pre_filename_stacked_data = ['', 'test_unroll_']
    
    X = [[None] * N_primitive for j in range(2)]
    Ct_target = [[None] * N_primitive for j in range(2)]
    normalized_phase_PSI_mult_phase_V = [[None] * N_primitive for j in range(2)]
    data_point_priority = [[None] * N_primitive for j in range(2)]
    
    for ntype in range(2):
        for np in range(N_primitive):
            [X[ntype][np],
             Ct_target[ntype][np],
             normalized_phase_PSI_mult_phase_V[ntype][np],
             data_point_priority[ntype][np]] = stackDataset(dataset_Ct,
                                                            subset_settings_indices,
                                                            mode_stack_dataset,
                                                            list_subset_outlier_ranked_demo_indices[ntype],
                                                            feature_type, np)
            assert (X[ntype][np].shape[0] == Ct_target[ntype][np].shape[0])
            assert (X[ntype][np].shape[0] == normalized_phase_PSI_mult_phase_V[ntype][np].shape[0])
            assert (X[ntype][np].shape[0] == data_point_priority[ntype][np].shape[0])
            
            if (os.path.isdir(out_data_dir)):
                sio.savemat((out_data_dir+'/'+list_pre_filename_stacked_data[ntype]+'prim_'+str(np)+'_X_'+feature_type+'_'+task_type+post_filename_stacked_data+'.mat'), X[ntype][np])
                sio.savemat((out_data_dir+'/'+list_pre_filename_stacked_data[ntype]+'prim_'+str(np)+'_Ct_target_'+task_type+post_filename_stacked_data+'.mat'), Ct_target[ntype][np])
                sio.savemat((out_data_dir+'/'+list_pre_filename_stacked_data[ntype]+'prim_'+str(np)+'_normalized_phase_PSI_mult_phase_V_'+task_type+post_filename_stacked_data+'.mat'), normalized_phase_PSI_mult_phase_V[ntype][np])
                sio.savemat((out_data_dir+'/'+list_pre_filename_stacked_data[ntype]+'prim_'+str(np)+'_data_point_priority_'+task_type+post_filename_stacked_data+'.mat'), data_point_priority[ntype][np])
            
            if (ntype == 0):
                print ('Total # of Data Points for Training            Primitive '+str(np)+': '+str(X[ntype][np].shape[0]))
            elif (ntype == 1):
                print ('Total # of Data Points for Generalization Test Primitive '+str(np)+': '+str(X[ntype][np].shape[0]))
    return X, Ct_target, normalized_phase_PSI_mult_phase_V, data_point_priority

def stackDataset(dataset, 
                 subset_settings_indices, 
                 mode, 
                 mode_arg, 
                 feature_type, 
                 primitive_no):
    assert ((mode == 1) or (mode == 2))
    
    if (mode == 1):
        # after ranking the trials based on the outlier metric
        # (ranked by dataset["outlier_metric"][primitive_no][setting_no] field,
        # i.e. rank 1==most likely is NOT an outlier; 
        # rank <end>==most likely is an outlier),
        # pick a subset of it, specified in subset_outlier_ranked_demo_indices, 
        # e.g. if subset_outlier_ranked_demo_indices=[1,3,4,5],
        # then this function will stack dataset of 
        # trials rank 1, 3, 4, and 5 (RECOMMENDED).
        
        subset_outlier_ranked_demo_indices = mode_arg
    elif (mode == 2):
        # pick trials with indices specified in subset_demos_index.
        
        subset_demos_indices = mode_arg
    
    N_settings_to_extract = len(subset_settings_indices)
    list_X_setting = [None] * N_settings_to_extract
    list_Ct_target_setting = [None] * N_settings_to_extract
    list_normalized_phase_PSI_mult_phase_V_setting = [None] * N_settings_to_extract
    list_data_point_priority_setting = [None] * N_settings_to_extract
    
    for ns_idx in range(N_settings_to_extract):
        setting_no = subset_settings_indices[ns_idx]
        if (mode == 1):
            existed_subset_outlier_ranked_demo_indices = list(set(range(len(dataset['trial_idx_ranked_by_outlier_metric_w_exclusion'][primitive_no][setting_no]))).intersection(set(subset_outlier_ranked_demo_indices)))
            subset_demos_indices = [dataset['trial_idx_ranked_by_outlier_metric_w_exclusion'][primitive_no][setting_no][ssidx] for ssidx in existed_subset_outlier_ranked_demo_indices]
        
        if (feature_type == 'raw'):
            list_X_setting[ns_idx] = np.hstack([dataset["sub_X"][primitive_no][setting_no][nd] for nd in subset_demos_indices])
        
        list_Ct_target_setting[ns_idx] = np.hstack([dataset["sub_Ct_target"][primitive_no][setting_no][nd] for nd in subset_demos_indices])
        if "sub_normalized_phase_PSI_mult_phase_V" in dataset:
            list_normalized_phase_PSI_mult_phase_V_setting[ns_idx] = np.hstack([dataset["sub_normalized_phase_PSI_mult_phase_V"][primitive_no][setting_no][nd] for nd in subset_demos_indices])
        if "sub_data_point_priority" in dataset:
            list_data_point_priority_setting[ns_idx] = np.hstack([dataset["sub_data_point_priority"][primitive_no][setting_no][nd] for nd in subset_demos_indices])
    
    X = np.hstack(list_X_setting).T
    Ct_target = np.hstack(list_Ct_target_setting).T
    if "sub_normalized_phase_PSI_mult_phase_V" in dataset:
        normalized_phase_PSI_mult_phase_V = np.hstack(list_normalized_phase_PSI_mult_phase_V_setting).T
    else:
        normalized_phase_PSI_mult_phase_V = None
    if "sub_data_point_priority" in dataset:
        data_point_priority = np.hstack(list_data_point_priority_setting)
        data_point_priority = data_point_priority.reshape(data_point_priority.shape[0],1)
    else:
        data_point_priority = None
    
    return X, Ct_target, normalized_phase_PSI_mult_phase_V, data_point_priority