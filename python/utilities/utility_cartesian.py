#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jul 24 10:00:00 2018
Modified on Dec 25 17:00:00 2018

@author: gsutanto
@comment: [1] Implemented analytical formula of Log() mappings of SO(3) and SE(3)
              and analytical formula of Exp() mappings of so(3) and se(3) 
              from "A Mathematical Introduction to Robotic Manipulation"
              textbook by Murray et al., page 413-414.
          [2] Tested and compared its results and computation time with 
              the general Matrix Logarithm (Log()) and Matrix Exponential (Exp())
              from Scipy Linear Algebra library.
"""

import time
import warnings as wa
import re
import numpy as np
import numpy.linalg as npla
import numpy.matlib as npma
import scipy.linalg as scla
import os
import sys
import copy

import utility_quaternion as util_quat


assert_epsilon = 1.0e-14
division_epsilon = 1.0e-100
symmetricity_epsilon = 1.0e-14

def getTensorDiag(tensor_input):
    assert (len(tensor_input.shape) == 3), "tensor_input has invalid number of dimensions!"
    assert (tensor_input.shape[1] == tensor_input.shape[2]), "tensor_input has to be square!!!"
    tensor_length = tensor_input.shape[0]
    tensor_input_dim = tensor_input.shape[1]
    tensor_diag = np.zeros((tensor_length, tensor_input_dim))
    for i in range(tensor_input_dim):
        tensor_diag[:,i] = tensor_input[:,i,i]
    return tensor_diag

def getTensorEye(tensor_length, tensor_dim):
    assert (tensor_length >= 0)
    assert (tensor_dim >= 1)
    if (tensor_length > 0):
        tensor_eye = np.zeros((tensor_length, tensor_dim, tensor_dim))
        for i in range(tensor_dim):
            tensor_eye[:,i,i] = np.ones(tensor_length)
        return tensor_eye
    else: # if (tensor_length == 0):
        return np.eye(tensor_dim)

def computeSkewSymmMatFromVec3(omega):
    assert ((len(omega.shape) >= 1) and (len(omega.shape) <= 2)), "omega has invalid number of dimensions!"
    if (len(omega.shape) == 1):
        omega = omega.reshape(1,3)
    assert (omega.shape[1] == 3)
    tensor_length = omega.shape[0]
    omegahat = np.zeros((tensor_length,3,3))
    sign_multiplier = -1
    for i in range(3):
        for j in range(i+1, 3):
            omegahat[:,i,j] = sign_multiplier * omega[:,3-i-j]
            omegahat[:,j,i] = -sign_multiplier * omega[:,3-i-j]
            sign_multiplier = -sign_multiplier
    if (tensor_length == 1):
        omegahat = omegahat[0,:,:]
    return omegahat

def computeVec3FromSkewSymmMat(omegahat, symm_epsilon=symmetricity_epsilon):
    assert ((len(omegahat.shape) >= 2) and (len(omegahat.shape) <= 3)), "omegahat has invalid number of dimensions!"
    if (len(omegahat.shape) == 2):
        omegahat = omegahat.reshape(1,3,3)
    assert (omegahat.shape[1] == 3)
    assert (omegahat.shape[2] == 3)
    assert (npla.norm(getTensorDiag(omegahat), ord=2, axis=1) < assert_epsilon).all(), ("omegahat = \n" + str(omegahat))
    for i in range(3):
        for j in range(i+1, 3):
            v1 = omegahat[:,i,j]
            v2 = omegahat[:,j,i]
            err = np.fabs(v1 + v2)
            assert (err < symm_epsilon).all(), ("There is err >= %f = symm_epsilon; err=\n"%symm_epsilon + str(err))
    tensor_length = omegahat.shape[0]
    omega = np.zeros((tensor_length, 3))
    omega[:,0] = 0.5 * (omegahat[:,2,1] - omegahat[:,1,2])
    omega[:,1] = 0.5 * (omegahat[:,0,2] - omegahat[:,2,0])
    omega[:,2] = 0.5 * (omegahat[:,1,0] - omegahat[:,0,1])
    if (tensor_length == 1):
        omega = omega[0,:]
    return omega

def computeKseehatFromWrench(wrench):
    assert ((len(wrench.shape) >= 1) and (len(wrench.shape) <= 2)), "wrench has invalid number of dimensions!"
    if (len(wrench.shape) == 1):
        wrench = wrench.reshape(1,6)
    assert (wrench.shape[1] == 6)
    v = wrench[:,:3]
    omega = wrench[:,3:6]
    tensor_length = wrench.shape[0]
    omegahat = computeSkewSymmMatFromVec3(omega).reshape(tensor_length,3,3)
    kseehat = np.zeros((tensor_length,4,4))
    kseehat[:,:3,:3] = omegahat
    kseehat[:,:3,3] = v
    if (tensor_length == 1):
        kseehat = kseehat[0,:,:]
    return kseehat

def computeWrenchFromKseehat(kseehat, symm_epsilon=symmetricity_epsilon):
    assert ((len(kseehat.shape) >= 2) and (len(kseehat.shape) <= 3)), "kseehat has invalid number of dimensions!"
    if (len(kseehat.shape) == 2):
        kseehat = kseehat.reshape(1,4,4)
    assert (npla.norm(kseehat[:,3,:], ord=2, axis=1) < assert_epsilon).all(), ("kseehat = \n" + str(kseehat))
    tensor_length = kseehat.shape[0]
    v = kseehat[:,:3,3].reshape(tensor_length, 3)
    omegahat = kseehat[:,:3,:3]
    omega = computeVec3FromSkewSymmMat(omegahat, symm_epsilon).reshape(tensor_length, 3)
    wrench = np.hstack([v, omega])
    assert (wrench.shape[1] == 6), "wrench.shape[1] = %d" % wrench.shape[1]
    if (tensor_length == 1):
        wrench = wrench[0,:]
    return wrench

def computeRotationMatrixLogMap(R, div_epsilon=division_epsilon): # Conversion from SO(3) (R) to so(3) (omegahat)
    assert ((len(R.shape) >= 2) and (len(R.shape) <= 3)), "R has invalid number of dimensions!"
    if (len(R.shape) == 2):
        R = R.reshape(1,3,3)
    assert (R.shape[1] == 3)
    assert (R.shape[2] == 3)
    assert (np.fabs(npla.det(R) - 1.0) < assert_epsilon).all(), "det(R) = %f" % npla.det(R)
    tensor_length = R.shape[0]
    half_traceR_minus_one = (np.trace(R,axis1=1,axis2=2) - 1.0)/2.0
    if ((half_traceR_minus_one < -1.0).any()):
        wa.warn("Warning: There is half_traceR_minus_one < -1.0" + str(half_traceR_minus_one))
        half_traceR_minus_one_less_than_minus_one_idx = np.where(half_traceR_minus_one < -1.0)[0]
        half_traceR_minus_one[half_traceR_minus_one_less_than_minus_one_idx] = -1.0
    if ((half_traceR_minus_one > 1.0).any()):
        wa.warn("Warning: There is half_traceR_minus_one > 1.0" + str(half_traceR_minus_one))
        half_traceR_minus_one_greater_than_one_idx = np.where(half_traceR_minus_one > 1.0)[0]
        half_traceR_minus_one[half_traceR_minus_one_greater_than_one_idx] = 1.0
    
    theta = np.arccos(half_traceR_minus_one).reshape(tensor_length, 1, 1)
    omegahat = (R - R.transpose((0,2,1))) / np.tile(((2.0 * np.sin(theta)) + div_epsilon), (1, 3, 3))
    log_R_output = np.tile(theta, (1, 3, 3)) * omegahat
    if (tensor_length == 1):
        log_R_output = log_R_output[0,:,:]
    return log_R_output

def computeRotationMatrixExpMap(omegahat, symm_epsilon=symmetricity_epsilon, div_epsilon=division_epsilon): # Conversion from so(3) (omegahat) to SO(3) (R)
    assert ((len(omegahat.shape) >= 2) and (len(omegahat.shape) <= 3)), "omegahat has invalid number of dimensions!"
    if (len(omegahat.shape) == 2):
        omegahat = omegahat.reshape(1,3,3)
    assert (omegahat.shape[1] == 3)
    assert (omegahat.shape[2] == 3)
    tensor_length = omegahat.shape[0]
    omega = computeVec3FromSkewSymmMat(omegahat, symm_epsilon).reshape(tensor_length, 3)
    
    norm_omega = npla.norm(omega, ord=2, axis=1).reshape(tensor_length, 1, 1)
    exp_omegahat = (getTensorEye(tensor_length, 3) + 
                    (np.tile(((np.sin(norm_omega) + div_epsilon)/(norm_omega + div_epsilon)), (1, 3, 3)) * omegahat) + 
                    (np.tile(((1.0-np.cos(norm_omega))/(np.square(norm_omega + div_epsilon))), (1, 3, 3)) * np.matmul(omegahat, omegahat))
                    )
    if (tensor_length == 1):
        exp_omegahat = exp_omegahat[0,:,:]
    return exp_omegahat

def computeHomogeneousTransformationMatrixLogMap(T, symm_epsilon=symmetricity_epsilon, div_epsilon=division_epsilon): # Conversion from SE(3) (T) to se(3) (kseehat)
    assert ((len(T.shape) >= 2) and (len(T.shape) <= 3)), "T has invalid number of dimensions!"
    if (len(T.shape) == 2):
        T = T.reshape(1,4,4)
    assert (T.shape[1] == 4)
    assert (T.shape[2] == 4)
    assert (npla.norm(T[:,3,:3], ord=2, axis=1) < assert_epsilon).all()
    assert (np.fabs(T[:,3,3] - 1.0) < assert_epsilon).all()
    tensor_length = T.shape[0]
    R = T[:,:3,:3]
    omegahat = computeRotationMatrixLogMap(R, div_epsilon).reshape(tensor_length, 3, 3)
    omega = computeVec3FromSkewSymmMat(omegahat, symm_epsilon).reshape(tensor_length, 3)
    norm_omega = npla.norm(omega, ord=2, axis=1).reshape(tensor_length, 1, 1)
    
    Ainv = (getTensorEye(tensor_length, 3) 
            - (0.5*omegahat) 
            + (np.tile((((2.0*(np.sin(norm_omega) + div_epsilon))-((norm_omega + div_epsilon) * (1.0+np.cos(norm_omega))))
                        /(2*np.square(norm_omega + div_epsilon)*(np.sin(norm_omega) + div_epsilon))), (1, 3, 3)) 
               * np.matmul(omegahat, omegahat)))
    p = T[:,:3,3].reshape(tensor_length,3,1)
    kseehat = np.zeros((tensor_length,4,4))
    kseehat[:,:3,:3] = omegahat
    kseehat[:,:3,3] = np.matmul(Ainv, p).reshape(tensor_length,3)
    if (tensor_length == 1):
        kseehat = kseehat[0,:,:]
    return kseehat

def computeHomogeneousTransformationMatrixExpMap(kseehat, symm_epsilon=symmetricity_epsilon, div_epsilon=division_epsilon): # Conversion from se(3) (kseehat) to SE(3) (T)
    assert ((len(kseehat.shape) >= 2) and (len(kseehat.shape) <= 3)), "kseehat has invalid number of dimensions!"
    if (len(kseehat.shape) == 2):
        kseehat = kseehat.reshape(1,4,4)
    assert (kseehat.shape[1] == 4)
    assert (kseehat.shape[2] == 4)
    assert (npla.norm(kseehat[:,3,:], ord=2, axis=1) < assert_epsilon).all()
    tensor_length = kseehat.shape[0]
    omegahat = kseehat[:,:3,:3]
    exp_omegahat = computeRotationMatrixExpMap(omegahat, div_epsilon).reshape(tensor_length, 3, 3)
    omega = computeVec3FromSkewSymmMat(omegahat, symm_epsilon).reshape(tensor_length, 3)
    norm_omega = npla.norm(omega, ord=2, axis=1).reshape(tensor_length, 1, 1)
    
    A = (getTensorEye(tensor_length, 3)
         + (np.tile(((1.0-np.cos(norm_omega))/np.square(norm_omega+div_epsilon)), (1, 3, 3)) * omegahat)
         + (np.tile((((norm_omega + div_epsilon) - (np.sin(norm_omega) + div_epsilon))/
                     (np.square(norm_omega+div_epsilon)*(norm_omega+div_epsilon))), (1, 3, 3)) 
            * np.matmul(omegahat, omegahat))
         )
    v = kseehat[:,:3,3].reshape(tensor_length,3,1)
    exp_kseehat = getTensorEye(tensor_length, 4)
    exp_kseehat[:,:3,:3] = exp_omegahat
    exp_kseehat[:,:3,3] = np.matmul(A, v).reshape(tensor_length,3)
    if (tensor_length == 1):
        exp_kseehat = exp_kseehat[0,:,:]
    return exp_kseehat

def computeHomogeneousTransformMatrix(t, Q):
    assert ((len(t.shape) >= 1) and (len(t.shape) <= 2)), "t has invalid number of dimensions!"
    assert ((len(Q.shape) >= 1) and (len(Q.shape) <= 2)), "Q has invalid number of dimensions!"
    if ((len(t.shape) == 1) or ((t.shape[0] == 3) and (t.shape[1] == 1))):
        t = t.reshape(1,3)
    if ((len(Q.shape) == 1) or ((Q.shape[0] == 3) and (Q.shape[1] == 1))):
        Q = Q.reshape(1,4)
    assert (t.shape[0] == Q.shape[0]), ('The tensor length of t=%d and of Q=%d are mis-matched!' % (t.shape[0], Q.shape[0]))
    tensor_length = t.shape[0]
    log_Q = util_quat.computeQuaternionLogMap(Q)
    twice_log_Q = 2.0 * log_Q
    skew_symm_twice_log_Q = computeSkewSymmMatFromVec3(twice_log_Q)
    R = computeRotationMatrixExpMap(skew_symm_twice_log_Q)
    T = getTensorEye(tensor_length, 4)
    T[:,:3,:3] = R
    T[:,:3,3] = t
    if (tensor_length == 1):
        T = T[0,:,:]
    return T

def computeInverseHomogeneousTransformMatrix(T):
    assert ((len(T.shape) >= 2) and (len(T.shape) <= 3)), "T has invalid number of dimensions!"
    if (len(T.shape) == 2):
        T = T.reshape(1,4,4)
    assert (npla.norm(T[:,3,:3], ord=2, axis=1) < assert_epsilon).all()
    assert (np.fabs(T[:,3,3] - 1.0) < assert_epsilon).all()
    tensor_length = T.shape[0]
    R = T[:,:3,:3]
    assert (np.fabs(npla.det(R) - 1.0) < assert_epsilon).all(), "det(R) = %f" % npla.det(R)
    p = T[:,:3,3].reshape(tensor_length,3,1)
    Tinv = getTensorEye(tensor_length, 4)
    Rinv = R.transpose((0,2,1))
    pinv = -np.matmul(Rinv, p)
    Tinv[:,:3,:3] = Rinv
    Tinv[:,:3,3] = pinv.reshape(tensor_length,3)
    if (tensor_length == 1):
        Tinv = Tinv[0,:,:]
    return Tinv

def computeStackedNumpyLogM(M):
    assert (len(M.shape) == 3), "M has invalid number of dimensions!"
    npLogM_list = list()
    for M_idx in range(M.shape[0]):
        npLogM_list.append(scla.logm(M[M_idx,:,:]))
    npLogM = np.stack(npLogM_list)
    return npLogM

def computeStackedNumpyExpM(M):
    assert (len(M.shape) == 3), "M has invalid number of dimensions!"
    npExpM_list = list()
    for M_idx in range(M.shape[0]):
        npExpM_list.append(scla.expm(M[M_idx,:,:]))
    npExpM = np.stack(npExpM_list)
    return npExpM

def computeStackedNumpyInvM(M):
    assert (len(M.shape) == 3), "M has invalid number of dimensions!"
    npInvM_list = list()
    for M_idx in range(M.shape[0]):
        npInvM_list.append(npla.inv(M[M_idx,:,:]))
    npInvM = np.stack(npInvM_list)
    return npInvM


if __name__=='__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    diff_epsilon = 1.0e-10
    
    print ""
    print "SO(3) Log Mapping Test:"
    R1 = np.eye(3)
    print "R1 = \n", R1
    start = time.time()
    npLogR1 = scla.logm(R1)
    end = time.time()
    print "npLogR1 = \n", npLogR1
    print "Computation Time of npLogR1: ", (end - start)
    start = time.time()
    LogR1 = computeRotationMatrixLogMap(R1)
    end = time.time()
    print "AnalyticalLogR1 = \n", LogR1
    print "Computation Time of AnalyticalLogR1: ", (end - start)
    diff = npla.norm(npLogR1 - LogR1)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    theta2 = np.pi/2.0
    R2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(theta2), -np.sin(theta2)],
                   [0.0, np.sin(theta2), np.cos(theta2)]])
    print "R2 = \n", R2
    start = time.time()
    npLogR2 = scla.logm(R2)
    end = time.time()
    print "npLogR2 = \n", npLogR2
    print "Computation Time of npLogR2: ", (end - start)
    start = time.time()
    LogR2 = computeRotationMatrixLogMap(R2)
    end = time.time()
    print "AnalyticalLogR2 = \n", LogR2
    print "Computation Time of AnalyticalLogR2: ", (end - start)
    diff = npla.norm(npLogR2 - LogR2)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    theta3 = np.random.rand(1)[0]
    R3 = np.array([[np.cos(theta3), 0.0, np.sin(theta3)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(theta3), 0.0, np.cos(theta3)]])
    print "R3 = \n", R3
    start = time.time()
    npLogR3 = scla.logm(R3)
    end = time.time()
    print "npLogR3 = \n", npLogR3
    print "Computation Time of npLogR3: ", (end - start)
    start = time.time()
    LogR3 = computeRotationMatrixLogMap(R3)
    end = time.time()
    print "AnalyticalLogR3 = \n", LogR3
    print "Computation Time of AnalyticalLogR3: ", (end - start)
    diff = npla.norm(npLogR3 - LogR3)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    theta = np.random.rand(3)
    R4 = np.array([[[1.0, 0.0, 0.0],
                    [0.0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0.0, np.sin(theta[0]), np.cos(theta[0])]],
                   [[np.cos(theta[1]), 0.0, np.sin(theta[1])],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta[1]), 0.0, np.cos(theta[1])]],
                   [[np.cos(theta[2]), -np.sin(theta[2]), 0.0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0.0],
                    [0.0, 0.0, 1.0]],
                   [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]])
#    print "R4 = \n", R4
    start = time.time()
    npLogR4 = computeStackedNumpyLogM(R4)
    end = time.time()
#    print "npLogR4 = \n", npLogR4
    print "Computation Time of npLogR4: ", (end - start)
    start = time.time()
    LogR4 = computeRotationMatrixLogMap(R4)
    end = time.time()
#    print "AnalyticalLogR4 = \n", LogR4
    print "Computation Time of AnalyticalLogR4: ", (end - start)
    diff = npla.norm(npLogR4 - LogR4)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    print ""
    print "so(3) Exp Mapping Test:"
    omega1 = np.zeros(3,)
    omegahat1 = computeSkewSymmMatFromVec3(omega1)
    print "omega1 = \n", omega1
    print "omegahat1 = \n", omegahat1
    start = time.time()
    npExpomegahat1 = scla.expm(omegahat1)
    end = time.time()
    print "npExpomegahat1 = \n", npExpomegahat1
    print "Computation Time of npExpomegahat1: ", (end - start)
    start = time.time()
    Expomegahat1 = computeRotationMatrixExpMap(omegahat1)
    end = time.time()
    print "AnalyticalExpomegahat1 = \n", Expomegahat1
    print "Computation Time of AnalyticalExpomegahat1: ", (end - start)
    diff = npla.norm(npExpomegahat1 - Expomegahat1)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    omega2 = np.random.rand(3)
    omegahat2 = computeSkewSymmMatFromVec3(omega2)
    print "omega2 = \n", omega2
    print "omegahat2 = \n", omegahat2
    start = time.time()
    npExpomegahat2 = scla.expm(omegahat2)
    end = time.time()
    print "npExpomegahat2 = \n", npExpomegahat2
    print "Computation Time of npExpomegahat2: ", (end - start)
    start = time.time()
    Expomegahat2 = computeRotationMatrixExpMap(omegahat2)
    end = time.time()
    print "AnalyticalExpomegahat2 = \n", Expomegahat2
    print "Computation Time of AnalyticalExpomegahat2: ", (end - start)
    diff = npla.norm(npExpomegahat2 - Expomegahat2)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    omega3 = np.random.rand(7,3)
    omegahat3 = computeSkewSymmMatFromVec3(omega3)
    print "omega3 = \n", omega3
#    print "omegahat3 = \n", omegahat3
    start = time.time()
    npExpomegahat3 = computeStackedNumpyExpM(omegahat3)
    end = time.time()
#    print "npExpomegahat3 = \n", npExpomegahat3
    print "Computation Time of npExpomegahat3: ", (end - start)
    start = time.time()
    Expomegahat3 = computeRotationMatrixExpMap(omegahat3)
    end = time.time()
#    print "AnalyticalExpomegahat3 = \n", Expomegahat3
    print "Computation Time of AnalyticalExpomegahat3: ", (end - start)
    diff = npla.norm(npExpomegahat3 - Expomegahat3)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    print ""
    print "SE(3) Log Mapping Test:"
    T1 = np.eye(4)
    T1[:3,:3] = R1
    T1[:3,3] = np.random.rand(3)
    print "T1 = \n", T1
    start = time.time()
    npLogT1 = scla.logm(T1)
    end = time.time()
    print "npLogT1 = \n", npLogT1
    print "Computation Time of npLogT1: ", (end - start)
    start = time.time()
    LogT1 = computeHomogeneousTransformationMatrixLogMap(T1)
    end = time.time()
    print "AnalyticalLogT1 = \n", LogT1
    print "Computation Time of AnalyticalLogT1: ", (end - start)
    diff = npla.norm(npLogT1 - LogT1)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    T2 = np.eye(4)
    T2[:3,:3] = R2
    T2[:3,3] = np.random.rand(3)
    print "T2 = \n", T2
    start = time.time()
    npLogT2 = scla.logm(T2)
    end = time.time()
    print "npLogT2 = \n", npLogT2
    print "Computation Time of npLogT2: ", (end - start)
    start = time.time()
    LogT2 = computeHomogeneousTransformationMatrixLogMap(T2)
    end = time.time()
    print "AnalyticalLogT2 = \n", LogT2
    print "Computation Time of AnalyticalLogT2: ", (end - start)
    diff = npla.norm(npLogT2 - LogT2)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    T3 = getTensorEye(R4.shape[0],4)
    T3[:,:3,:3] = R4
    T3[:,:3,3] = np.random.rand(R4.shape[0],3)
#    print "T3 = \n", T3
    start = time.time()
    npLogT3 = computeStackedNumpyLogM(T3)
    end = time.time()
#    print "npLogT3 = \n", npLogT3
    print "Computation Time of npLogT3: ", (end - start)
    start = time.time()
    LogT3 = computeHomogeneousTransformationMatrixLogMap(T3)
    end = time.time()
#    print "AnalyticalLogT3 = \n", LogT3
    print "Computation Time of AnalyticalLogT3: ", (end - start)
    diff = npla.norm(npLogT3 - LogT3)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    print ""
    print "se(3) Exp Mapping Test:"
    kseehat1 = np.zeros((4,4))
    kseehat1[:3,:3] = omegahat1
    kseehat1[:3,3] = np.random.rand(3)
    print "kseehat1 = \n", kseehat1
    start = time.time()
    npExpkseehat1 = scla.expm(kseehat1)
    end = time.time()
    print "npExpkseehat1 = \n", npExpkseehat1
    print "Computation Time of npExpkseehat1: ", (end - start)
    start = time.time()
    Expkseehat1 = computeHomogeneousTransformationMatrixExpMap(kseehat1)
    end = time.time()
    print "AnalyticalExpkseehat1 = \n", Expkseehat1
    print "Computation Time of AnalyticalExpkseehat1: ", (end - start)
    diff = npla.norm(npExpkseehat1 - Expkseehat1)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    kseehat2 = np.zeros((4,4))
    kseehat2[:3,:3] = omegahat2
    kseehat2[:3,3] = np.random.rand(3)
    print "kseehat2 = \n", kseehat2
    start = time.time()
    npExpkseehat2 = scla.expm(kseehat2)
    end = time.time()
    print "npExpkseehat2 = \n", npExpkseehat2
    print "Computation Time of npExpkseehat2: ", (end - start)
    start = time.time()
    Expkseehat2 = computeHomogeneousTransformationMatrixExpMap(kseehat2)
    end = time.time()
    print "AnalyticalExpkseehat2 = \n", Expkseehat2
    print "Computation Time of AnalyticalExpkseehat2: ", (end - start)
    diff = npla.norm(npExpkseehat2 - Expkseehat2)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    kseehat3 = np.zeros((omegahat3.shape[0],4,4))
    kseehat3[:,:3,:3] = omegahat3
    kseehat3[:,:3,3] = np.random.rand(omegahat3.shape[0],3)
#    print "kseehat3 = \n", kseehat3
    start = time.time()
    npExpkseehat3 = computeStackedNumpyExpM(kseehat3)
    end = time.time()
#    print "npExpkseehat3 = \n", npExpkseehat3
    print "Computation Time of npExpkseehat3: ", (end - start)
    start = time.time()
    Expkseehat3 = computeHomogeneousTransformationMatrixExpMap(kseehat3)
    end = time.time()
#    print "AnalyticalExpkseehat3 = \n", Expkseehat3
    print "Computation Time of AnalyticalExpkseehat3: ", (end - start)
    diff = npla.norm(npExpkseehat3 - Expkseehat3)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    wrench4 = np.random.rand(10,6)
    kseehat4 = computeKseehatFromWrench(wrench4)
    print "wrench4 = \n", wrench4
#    print "kseehat4 = \n", kseehat4
    start = time.time()
    npExpkseehat4 = computeStackedNumpyExpM(kseehat4)
    end = time.time()
#    print "npExpkseehat4 = \n", npExpkseehat4
    print "Computation Time of npExpkseehat4: ", (end - start)
    start = time.time()
    Expkseehat4 = computeHomogeneousTransformationMatrixExpMap(kseehat4)
    end = time.time()
#    print "AnalyticalExpkseehat4 = \n", Expkseehat4
    print "Computation Time of AnalyticalExpkseehat4: ", (end - start)
    diff = npla.norm(npExpkseehat4 - Expkseehat4)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    print ""
    print "Inverse Homogeneous Transformation Matrix Test:"
    print "T1 = \n", T1
    start = time.time()
    npInvT1 = npla.inv(T1)
    end = time.time()
    print "npInvT1 = \n", npInvT1
    print "Computation Time of npInvT1: ", (end - start)
    start = time.time()
    InvT1 = computeInverseHomogeneousTransformMatrix(T1)
    end = time.time()
    print "AnalyticalInvT1 = \n", InvT1
    print "Computation Time of AnalyticalInvT1: ", (end - start)
    diff = npla.norm(npInvT1 - InvT1)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    print "T2 = \n", T2
    start = time.time()
    npInvT2 = npla.inv(T2)
    end = time.time()
    print "npInvT2 = \n", npInvT2
    print "Computation Time of npInvT2: ", (end - start)
    start = time.time()
    InvT2 = computeInverseHomogeneousTransformMatrix(T2)
    end = time.time()
    print "AnalyticalInvT2 = \n", InvT2
    print "Computation Time of AnalyticalInvT2: ", (end - start)
    diff = npla.norm(npInvT2 - InvT2)
    assert diff < diff_epsilon, "diff = %e" % diff
    
#    print "T3 = \n", T3
    start = time.time()
    npInvT3 = computeStackedNumpyInvM(T3)
    end = time.time()
#    print "npInvT3 = \n", npInvT3
    print "Computation Time of npInvT3: ", (end - start)
    start = time.time()
    InvT3 = computeInverseHomogeneousTransformMatrix(T3)
    end = time.time()
#    print "AnalyticalInvT3 = \n", InvT3
    print "Computation Time of AnalyticalInvT3: ", (end - start)
    diff = npla.norm(npInvT3 - InvT3)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    T4 = Expkseehat4
#    print "T4 = \n", T4
    start = time.time()
    npInvT4 = computeStackedNumpyInvM(T4)
    end = time.time()
#    print "npInvT4 = \n", npInvT4
    print "Computation Time of npInvT4: ", (end - start)
    start = time.time()
    InvT4 = computeInverseHomogeneousTransformMatrix(T4)
    end = time.time()
#    print "AnalyticalInvT4 = \n", InvT4
    print "Computation Time of AnalyticalInvT4: ", (end - start)
    diff = npla.norm(npInvT4 - InvT4)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    print ""
    print "SO(3) Cubed/Composition Test:"
    print "R3 = \n", R3
    start = time.time()
    R3cubed = np.matmul(R3, np.matmul(R3, R3))
    end = time.time()
    print "R3cubed = \n", R3cubed
    print "Computation Time of R3cubed: ", (end - start)
    start = time.time()
    exp_3Xlog_R3 = computeRotationMatrixExpMap(3 * LogR3)
    end = time.time()
    print "exp_3Xlog_R3 = \n", exp_3Xlog_R3
    print "Computation Time of exp_3Xlog_R3: ", (end - start)
    diff = npla.norm(R3cubed - exp_3Xlog_R3)
    assert diff < diff_epsilon, "diff = %e" % diff
    
#    print "R4 = \n", R4
    start = time.time()
    R4cubed = np.matmul(R4, np.matmul(R4, R4))
    end = time.time()
#    print "R4cubed = \n", R4cubed
    print "Computation Time of R4cubed: ", (end - start)
    start = time.time()
    exp_3Xlog_R4 = computeRotationMatrixExpMap(3 * LogR4)
    end = time.time()
#    print "exp_3Xlog_R4 = \n", exp_3Xlog_R4
    print "Computation Time of exp_3Xlog_R4: ", (end - start)
    diff = npla.norm(R4cubed - exp_3Xlog_R4)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    print ""
    print "SE(3) Cubed/Composition Test:"
    print "T2 = \n", T2
    start = time.time()
    T2cubed = np.matmul(T2, np.matmul(T2, T2))
    end = time.time()
    print "T2cubed = \n", T2cubed
    print "Computation Time of T2cubed: ", (end - start)
    start = time.time()
    exp_3Xlog_T2 = computeHomogeneousTransformationMatrixExpMap(3 * LogT2)
    end = time.time()
    print "exp_3Xlog_T2 = \n", exp_3Xlog_T2
    print "Computation Time of exp_3Xlog_T2: ", (end - start)
    diff = npla.norm(T2cubed - exp_3Xlog_T2)
    assert diff < diff_epsilon, "diff = %e" % diff
    
#    print "T3 = \n", T3
    start = time.time()
    T3cubed = np.matmul(T3, np.matmul(T3, T3))
    end = time.time()
#    print "T3cubed = \n", T3cubed
    print "Computation Time of T3cubed: ", (end - start)
    start = time.time()
    exp_3Xlog_T3 = computeHomogeneousTransformationMatrixExpMap(3 * LogT3)
    end = time.time()
#    print "exp_3Xlog_T3 = \n", exp_3Xlog_T3
    print "Computation Time of exp_3Xlog_T3: ", (end - start)
    diff = npla.norm(T3cubed - exp_3Xlog_T3)
    assert diff < diff_epsilon, "diff = %e" % diff
    
#    print "T4 = \n", T4
    start = time.time()
    log_T4 = computeHomogeneousTransformationMatrixLogMap(T4)
    wrench_log_T4 = computeWrenchFromKseehat(log_T4)
    end = time.time()
    print "wrench4 = \n", wrench4
    print "wrench_log_T4 = \n", wrench_log_T4
    print "Computation Time of wrench_log_T4: ", (end - start)
    diff = npla.norm(wrench4 - wrench_log_T4)
    assert diff < diff_epsilon, "diff = %e" % diff
    
    ## The following is NOT the same (incorrect algebra; NOT commutative)
#    print ""
#    print "SO(3) Composition Test:"
#    print "R2 = \n", R2
#    print "R3 = \n", R3
#    start = time.time()
#    R2composeR3 = np.matmul(R2, R3)
#    end = time.time()
#    print "R2composeR3 = \n", R2composeR3
#    print "Computation Time of R2composeR3: ", (end - start)
#    start = time.time()
#    exp_logR2_plus_logR3 = computeRotationMatrixExpMap(LogR2 + LogR3)
#    end = time.time()
#    print "exp_logR2_plus_logR3 = \n", exp_logR2_plus_logR3
#    print "Computation Time of exp_logR2_plus_logR3: ", (end - start)
#    diff = npla.norm(R2composeR3 - exp_logR2_plus_logR3)
#    assert diff < diff_epsilon, "diff = %e" % diff
    
    ## The following is NOT the same (incorrect algebra; NOT commutative)
#    print ""
#    print "SE(3) Composition Test:"
#    print "T1 = \n", T1
#    print "T2 = \n", T2
#    start = time.time()
#    T1composeT2 = np.matmul(T1, T2)
#    end = time.time()
#    print "T1composeT2 = \n", T1composeT2
#    print "Computation Time of T1composeT2: ", (end - start)
#    start = time.time()
#    exp_logT1_plus_logT2 = computeHomogeneousTransformationMatrixExpMap(LogT1 + LogT2)
#    end = time.time()
#    print "exp_logT1_plus_logT2 = \n", exp_logT1_plus_logT2
#    print "Computation Time of exp_logT1_plus_logT2: ", (end - start)
#    diff = npla.norm(T1composeT2 - exp_logT1_plus_logT2)
#    assert diff < diff_epsilon, "diff = %e" % diff

    t = np.array([[1,2,3.0],[4,5,6],[7,8,9],[101,103,105],[201,53,405]])
    Q = np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0],[0.7073883, 0, 0.4998009, 0.4998009]])
    T = computeHomogeneousTransformMatrix(t, Q)
    print "T = \n", T