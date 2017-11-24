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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from DMPTrajectory import *
from DMPState import *
from utilities import *

PARALLEL_VECTOR_PROJECTION_THRESHOLD = 0.9
MIN_CTRAJ_LOCAL_COORD_OPTION_NO = 0
MAX_CTRAJ_LOCAL_COORD_OPTION_NO = 3
NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS = 5

# CartCoordDMPLocalCoordinateFrameMethod
NO_LOCAL_COORD_FRAME = 0
GSUTANTO_LOCAL_COORD_FRAME = 1
SCHAAL_LOCAL_COORD_FRAME = 2
KROEMER_LOCAL_COORD_FRAME = 3

class CartesianCoordTransformer:
    'Class for coordinate transformations of discrete Cartesian Coordinate DMPs.'
    
    def __init__(self, name=""):
        self.name = name
        # the followings are only relevant for GSUTANTO_LOCAL_COORD_FRAME selection:
        self.theta_kernel_centers = np.zeros((NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1))
        self.theta_kernel_Ds = np.zeros((NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1))
        self.basis_vector_weights_unnormalized = np.zeros((NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1))
        self.basis_matrix = np.zeros((3, NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS))
    
    def isValid(self):
        # the followings are only relevant for GSUTANTO_LOCAL_COORD_FRAME selection:
        assert (self.theta_kernel_centers.shape == (NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1))
        assert (self.theta_kernel_Ds.shape == (NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1))
        assert (self.basis_vector_weights_unnormalized.shape == (NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1))
        assert (self.basis_matrix.shape == (3,NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS))
        return True
    
    def computeCTrajCoordTransforms(self, cart_dmptrajectory, ctraj_local_coordinate_frame_selection):
        assert (self.isValid())
        cart_dmptrajectory_length = cart_dmptrajectory.getLength()
        assert (cart_dmptrajectory_length >= 2)
        assert (((ctraj_local_coordinate_frame_selection == KROEMER_LOCAL_COORD_FRAME) and 
                 (cart_dmptrajectory_length < 3)) == False)
        assert (cart_dmptrajectory.isValid())
        assert (cart_dmptrajectory.dmp_num_dimensions == 3)
        assert ((ctraj_local_coordinate_frame_selection >= MIN_CTRAJ_LOCAL_COORD_OPTION_NO) and (ctraj_local_coordinate_frame_selection <= MAX_CTRAJ_LOCAL_COORD_OPTION_NO))
        
        start_cart_position = cart_dmptrajectory.getDMPStateAtIndex(0).getX()
        goal_cart_position = cart_dmptrajectory.getDMPStateAtIndex(cart_dmptrajectory_length-1).getX()
        
        translation_vector = start_cart_position
        
        global_x_axis = np.array([[1.0],[0.0],[0.0]])
        global_y_axis = np.array([[0.0],[1.0],[0.0]])
        global_z_axis = np.array([[0.0],[0.0],[1.0]])
        
        if (ctraj_local_coordinate_frame_selection == NO_LOCAL_COORD_FRAME):
            local_x_axis = global_x_axis
            local_y_axis = global_y_axis
            local_z_axis = global_z_axis
            translation_vector = np.zeros((3,1))
        elif (ctraj_local_coordinate_frame_selection == GSUTANTO_LOCAL_COORD_FRAME):
            local_x_axis = goal_cart_position - start_cart_position
            local_x_axis = local_x_axis/np.linalg.norm(local_x_axis)
            
            # theta is the angle formed between global +z axis and local +x axis:
            cos_theta = np.matmul(local_x_axis.T,global_z_axis)[0,0]
            theta = np.abs(np.arccos(cos_theta))
            
            # compute the kernel centers and spread/standard deviation (D):
            self.theta_kernel_centers = np.linspace(0,np.pi,NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS).reshape(NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1)
            self.theta_kernel_Ds = np.square(np.diff(self.theta_kernel_centers,axis=0)*0.55)
            self.theta_kernel_Ds = 1.0/np.vstack((self.theta_kernel_Ds,self.theta_kernel_Ds[-1,0]))
            
            # compute the kernel/basis vector weights:
            self.basis_vector_weights_unnormalized = np.exp(-0.5 * np.square((theta*np.ones((NGSUTANTO_LOCAL_COORD_FRAMEBASIS_VECTORS,1)))-self.theta_kernel_centers) * self.theta_kernel_Ds)
            
            # theta == 0 (local_x_axis is perfectly aligned with global_z_axis) corresponds to anchor_local_z_axis = global_y_axis:
            self.basis_matrix[:,0] = global_y_axis.reshape(3,)
            # theta == pi/4 corresponds to anchor_local_z_axis = global_z_axis X local_x_axis:
            self.basis_matrix[:,1] = np.cross(global_z_axis.reshape(3,), local_x_axis.reshape(3,))
            # theta == pi/2 (local_x_axis is perpendicular from global_z_axis) corresponds to anchor_local_z_axis = global_z_axis:
            self.basis_matrix[:,2] = global_z_axis.reshape(3,)
            # theta == 3*pi/4 corresponds to anchor_local_z_axis = -global_z_axis X local_x_axis:
            self.basis_matrix[:,3] = -np.cross(global_z_axis.reshape(3,), local_x_axis.reshape(3,))
            # theta == pi (local_x_axis is perfectly aligned with -global_z_axis) corresponds to anchor_local_z_axis = -global_y_axis:
            self.basis_matrix[:,4] = -global_y_axis.reshape(3,)
            
            # anchor_local_z_axis are the normalized weighted combination of the basis vectors:
            anchor_local_z_axis = np.matmul(self.basis_matrix, self.basis_vector_weights_unnormalized)/np.sum(self.basis_vector_weights_unnormalized)
            anchor_local_z_axis = anchor_local_z_axis/np.linalg.norm(anchor_local_z_axis)
            
            local_y_axis = np.cross(anchor_local_z_axis.reshape(3,), local_x_axis.reshape(3,)).reshape(3,1)
            local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
            local_z_axis = np.cross(local_x_axis.reshape(3,), local_y_axis.reshape(3,)).reshape(3,1)
            local_z_axis = local_z_axis/np.linalg.norm(local_z_axis)
        elif (ctraj_local_coordinate_frame_selection == SCHAAL_LOCAL_COORD_FRAME):
            # =============================================================================
            # [IMPORTANT] Notice the IF-ELSE block below:
            # >> IF   block: condition handled: local x-axis is "NOT TOO parallel" to global z-axis
            #                                   (checked by: if dot-product or projection length is below the threshold)
            #                handling:          local x-axis and global z-axis (~= local z-axis) become the anchor axes,
            #                                   local y-axis is the cross-product between them (z X x)
            # >> ELSE block: condition handled: local x-axis is "TOO parallel" to global z-axis
            #                                   (checked by: if dot-product or projection length is above the threshold)
            #                handling:          local x-axis and global y-axis (~= local y-axis) become the anchor axes,
            #                                   local z-axis is the cross-product between them (x X y)
            # REMEMBER THESE ASSUMPTIONS!!! as this may introduce a potential problems at some conditions in the future...
            # To avoid problems, during trajectory demonstrations, please AVOID demonstrating trajectories
            # whose start and end points are having very close global z-coordinates (i.e. "vertical" trajectories)!!!
            # =============================================================================
            local_x_axis = goal_cart_position - start_cart_position
            local_x_axis = local_x_axis/np.linalg.norm(local_x_axis)
            # check if local x-axis is "too parallel" with the global z-axis; if not:
            if (np.abs(np.matmul(local_x_axis.T,global_z_axis)[0,0]) <= PARALLEL_VECTOR_PROJECTION_THRESHOLD):
                local_z_axis = global_z_axis
                local_y_axis = np.cross(local_z_axis.reshape(3,), local_x_axis.reshape(3,)).reshape(3,1)
                local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
                local_z_axis = np.cross(local_x_axis.reshape(3,), local_y_axis.reshape(3,)).reshape(3,1)
                local_z_axis = local_z_axis/np.linalg.norm(local_z_axis)
            else: # if it is "parallel enough":
                local_y_axis = global_y_axis
                local_z_axis = np.cross(local_x_axis.reshape(3,), local_y_axis.reshape(3,)).reshape(3,1)
                local_z_axis = local_z_axis/np.linalg.norm(local_z_axis)
                local_y_axis = np.cross(local_z_axis.reshape(3,), local_x_axis.reshape(3,)).reshape(3,1)
                local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
        elif (ctraj_local_coordinate_frame_selection == KROEMER_LOCAL_COORD_FRAME):
            # =============================================================================
            # See O.B. Kroemer, R. Detry, J. Piater, and J. Peters;
            #     Combining active learning and reactive control for robot grasping;
            #     Robotics and Autonomous Systems 58 (2010), page 1111 (Figure 5)
            # See the definition of CartesianCoordDMP in CartesianCoordDMP.h (C++ Source Code).
            # =============================================================================
            approaching_goal_cart_position = cart_dmptrajectory.getDMPStateAtIndex(cart_dmptrajectory_length-2).getX()
            local_x_axis = goal_cart_position - approaching_goal_cart_position
            local_x_axis = local_x_axis/np.linalg.norm(local_x_axis)
            local_y_axis = goal_cart_position - start_cart_position
            local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
            # check if local x-axis is "too parallel" with the "temporary" local y-axis; if not:
            if (np.abs(np.matmul(local_x_axis.T,local_y_axis)[0,0]) <= PARALLEL_VECTOR_PROJECTION_THRESHOLD):
                local_z_axis = np.cross(local_x_axis.reshape(3,), local_y_axis.reshape(3,)).reshape(3,1)
                local_z_axis = local_z_axis/np.linalg.norm(local_z_axis)
                local_y_axis = np.cross(local_z_axis.reshape(3,), local_x_axis.reshape(3,)).reshape(3,1)
                local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
            else: # if it is "parallel enough" (following the same method as Schaal's trajectory local coordinate system formulation):
                if (np.abs(np.matmul(local_x_axis.T,global_z_axis)[0,0]) <= PARALLEL_VECTOR_PROJECTION_THRESHOLD):
                    local_z_axis = global_z_axis
                    local_y_axis = np.cross(local_z_axis.reshape(3,), local_x_axis.reshape(3,)).reshape(3,1)
                    local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
                    local_z_axis = np.cross(local_x_axis.reshape(3,), local_y_axis.reshape(3,)).reshape(3,1)
                    local_z_axis = local_z_axis/np.linalg.norm(local_z_axis)
                else: # if it is "parallel enough":
                    local_y_axis = global_y_axis
                    local_z_axis = np.cross(local_x_axis.reshape(3,), local_y_axis.reshape(3,)).reshape(3,1)
                    local_z_axis = local_z_axis/np.linalg.norm(local_z_axis)
                    local_y_axis = np.cross(local_z_axis.reshape(3,), local_x_axis.reshape(3,)).reshape(3,1)
                    local_y_axis = local_y_axis/np.linalg.norm(local_y_axis)
        
        rotation_matrix = np.hstack([local_x_axis, local_y_axis, local_z_axis])
        
        rel_homogen_transform_matrix_local_to_global = np.eye(4)
        rel_homogen_transform_matrix_local_to_global[0:3,0:3] = rotation_matrix
        rel_homogen_transform_matrix_local_to_global[0:3,3] = translation_vector.reshape(3,)
        
        rel_homogen_transform_matrix_global_to_local = np.eye(4)
        rel_homogen_transform_matrix_global_to_local[0:3,0:3] = rotation_matrix.T
        rel_homogen_transform_matrix_global_to_local[0:3,3] = np.matmul(-rotation_matrix.T, translation_vector).reshape(3,)
        
        return rel_homogen_transform_matrix_local_to_global, rel_homogen_transform_matrix_global_to_local
    
    def computeCTrajAtNewCoordSys(self, cart_dmptrajectory_old, rel_homogen_transform_matrix_old_to_new):
        assert (self.isValid())
        assert (np.count_nonzero(rel_homogen_transform_matrix_old_to_new) > 0)
        cart_dmptrajectory_length = cart_dmptrajectory_old.getLength()
        assert (cart_dmptrajectory_length >= 1)
        
        X_new = np.matmul(rel_homogen_transform_matrix_old_to_new, np.vstack([cart_dmptrajectory_old.getX(), np.ones((1,cart_dmptrajectory_length))]))[0:3,:]
        Xd_new = np.matmul(rel_homogen_transform_matrix_old_to_new[0:3,0:3], cart_dmptrajectory_old.getXd())
        Xdd_new = np.matmul(rel_homogen_transform_matrix_old_to_new[0:3,0:3], cart_dmptrajectory_old.getXdd())
        time_new = cart_dmptrajectory_old.getTime()
        if (cart_dmptrajectory_length == 1):
            cart_dmptrajectory_new = DMPState(X_new, Xd_new, Xdd_new, time_new)
        elif (cart_dmptrajectory_length > 1):
            cart_dmptrajectory_new = DMPTrajectory(X_new, Xd_new, Xdd_new, time_new)
        return cart_dmptrajectory_new
    
    def convertCTrajAtOldToNewCoordSys(self, cart_dmptrajectory_old, rel_homogen_transform_matrix_old_to_new):
        return self.computeCTrajAtNewCoordSys(cart_dmptrajectory_old, rel_homogen_transform_matrix_old_to_new)
    
    def computeCVecAtNewCoordSys(self, cart_vector_old, rel_homogen_transform_matrix_old_to_new):
        assert (self.isValid())
        assert (cart_vector_old.shape[1] >= 1), 'cart_vector_old.shape[1] = ' + str(cart_vector_old.shape[1])
        assert ((cart_vector_old.shape[0] == 3) or (cart_vector_old.shape[0] == 4)), 'cart_vector_old.shape[0] = ' + str(cart_vector_old.shape[0])
        if (cart_vector_old.shape[0] == 4):
            assert (np.array_equal(cart_vector_old[3,:].reshape(1,cart_vector_old.shape[1]), 
                                   np.ones((1,cart_vector_old.shape[1])))), 'Input is NOT a valid homogeneous vector(s)!'
        assert (np.count_nonzero(rel_homogen_transform_matrix_old_to_new) > 0)
        
        if (cart_vector_old.shape[0] == 3):
            cart_vector_new = np.matmul(rel_homogen_transform_matrix_old_to_new[0:3,0:3], cart_vector_old)
        elif (cart_vector_old.shape[0] == 4):
            cart_vector_new = np.matmul(rel_homogen_transform_matrix_old_to_new, cart_vector_old)
            assert (np.array_equal(cart_vector_new[3,:].reshape(1,cart_vector_new.shape[1]), 
                                   np.ones((1,cart_vector_new.shape[1])))), 'Output is NOT a valid homogeneous vector(s)!'
        
        return cart_vector_new
    
    def computeCPosAtNewCoordSys(self, pos_3D_old, rel_homogen_transform_matrix_old_to_new):
        pos_3D_H_old = np.vstack([pos_3D_old,np.ones((1,pos_3D_old.shape[1]))])
        return self.computeCVecAtNewCoordSys(pos_3D_H_old, rel_homogen_transform_matrix_old_to_new)