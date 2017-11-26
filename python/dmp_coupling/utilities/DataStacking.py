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

def prepareData(task_type, dataset_Ct, subset_settings_indices,
                considered_subset_outlier_ranked_demo_indices,
                generalization_subset_outlier_ranked_demo_indices,
                post_filename_stacked_data,
                out_data_dir):
    mode_stack_dataset = 1
    
    N_primitive = len(dataset_Ct["sub_Ct_target"])
    
    training_subset_outlier_ranked_demo_indices = list(set(considered_subset_outlier_ranked_demo_indices) - set(generalization_subset_outlier_ranked_demo_indices))
    if (generalization_subset_outlier_ranked_demo_indices == []):
        generalization_subset_outlier_ranked_demo_indices = [1] # CANNOT really be empty (for further Python processing)
    
    list_subset_outlier_ranked_demo_indices = [training_subset_outlier_ranked_demo_indices, 
                                               generalization_subset_outlier_ranked_demo_indices]
    list_pre_filename_stacked_data = ['', 'test_unroll_']
    
    for ntype in range(2):
        for np in range(N_primitive):
            
    return None

def stackDataset():
    return None