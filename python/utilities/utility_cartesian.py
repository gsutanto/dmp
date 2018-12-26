#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jul 24 10:00:00 2018
Modified on Dec 25 17:00:00 2018

@author: gsutanto
@comment: implemented from "A Mathematical Introduction to Robotic Manipulation"
          textbook by Murray et al., page 413-414
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
denominator_epsilon = 1.0e-30
symmetricity_epsilon = 1.0e-14

def getTensorDiag(tensor_input):
    assert (tensor_input.shape[1] == tensor_input.shape[2]), "tensor_input has to be square!!!"
    tensor_length = tensor_input.shape[0]
    tensor_input_dim = tensor_input.shape[1]
    tensor_diag = np.zeros((tensor_length, tensor_input_dim))
    for i in range(tensor_input_dim):
        tensor_diag[:,i] = tensor_input[:,i,i]
    return tensor_diag

def getTensorEye(tensor_length, tensor_dim):
    tensor_eye = np.zeros((tensor_length, tensor_dim, tensor_dim))
    for i in range(tensor_dim):
        tensor_eye[:,i,i] = np.ones(tensor_length)
    return tensor_eye

def computeSkewSymmMatFromVec3(omega):
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
    if (len(kseehat.shape) == 1):
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

def computeRotationMatrixLogMap(R, denom_epsilon=denominator_epsilon): # Conversion from SO(3) (R) to so(3) (omegahat)
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
    omegahat = (R - R.transpose((0,2,1))) / np.tile(((2.0 * np.sin(theta)) + denom_epsilon), (1, 3, 3))
    log_R_output = np.tile(theta, (1, 3, 3)) * omegahat
    if (tensor_length == 1):
        log_R_output = log_R_output[0,:,:]
    return log_R_output

def computeRotationMatrixExpMap(omegahat, symm_epsilon=symmetricity_epsilon, denom_epsilon=denominator_epsilon): # Conversion from so(3) (omegahat) to SO(3) (R)
    if (len(omegahat.shape) == 2):
        omegahat = omegahat.reshape(1,3,3)
    assert (omegahat.shape[1] == 3)
    assert (omegahat.shape[2] == 3)
    tensor_length = omegahat.shape[0]
    omega = computeVec3FromSkewSymmMat(omegahat, symm_epsilon).reshape(tensor_length, 3)
    
    norm_omega = npla.norm(omega, ord=2, axis=1).reshape(tensor_length, 1, 1)
    exp_omegahat = (getTensorEye(tensor_length, 3) + 
                    (np.tile((np.sin(norm_omega)/(norm_omega + denom_epsilon)), (1, 3, 3)) * omegahat) + 
                    (np.tile(((1.0-np.cos(norm_omega))/(np.square(norm_omega + denom_epsilon))), (1, 3, 3)) * np.matmul(omegahat, omegahat))
                    )
    if (tensor_length == 1):
        exp_omegahat = exp_omegahat[0,:,:]
    return exp_omegahat

def computeHomogeneousTransformationMatrixLogMap(T, symm_epsilon=symmetricity_epsilon, denom_epsilon=denominator_epsilon): # Conversion from SE(3) (T) to se(3) (kseehat)
    if (len(T.shape) == 2):
        T = T.reshape(1,4,4)
    assert (T.shape[1] == 4)
    assert (T.shape[2] == 4)
    assert (npla.norm(T[:,3,:3], ord=2, axis=1) < assert_epsilon).all()
    assert (np.fabs(T[:,3,3] - 1.0) < assert_epsilon).all()
    tensor_length = T.shape[0]
    R = T[:,:3,:3]
    omegahat = computeRotationMatrixLogMap(R, denom_epsilon).reshape(tensor_length, 3, 3)
    omega = computeVec3FromSkewSymmMat(omegahat, symm_epsilon).reshape(tensor_length, 3)
    norm_omega = npla.norm(omega, ord=2, axis=1).reshape(tensor_length, 1, 1)
    
    Ainv = (getTensorEye(tensor_length, 3) 
            - (0.5*omegahat) 
            + (np.tile((((2.0*np.sin(norm_omega))-(norm_omega*(1.0+np.cos(norm_omega))))
                        /((2*np.square(norm_omega)*np.sin(norm_omega))+denom_epsilon)), (1, 3, 3)) 
               * np.matmul(omegahat, omegahat)))
    p = T[:,:3,3].reshape(tensor_length,3,1)
    kseehat = np.zeros((tensor_length,4,4))
    kseehat[:,:3,:3] = omegahat
    kseehat[:,:3,3] = np.matmul(Ainv, p).reshape(tensor_length,3)
    if (tensor_length == 1):
        kseehat = kseehat[0,:,:]
    return kseehat

def computeHomogeneousTransformationMatrixExpMap(kseehat, symm_epsilon=symmetricity_epsilon, denom_epsilon=denominator_epsilon): # Conversion from se(3) (kseehat) to SE(3) (T)
    if (len(kseehat.shape) == 2):
        kseehat = kseehat.reshape(1,4,4)
    assert (kseehat.shape[1] == 4)
    assert (kseehat.shape[2] == 4)
    assert (npla.norm(kseehat[:,3,:], ord=2, axis=1) < assert_epsilon).all()
    tensor_length = kseehat.shape[0]
    omegahat = kseehat[:,:3,:3]
    exp_omegahat = computeRotationMatrixExpMap(omegahat, denom_epsilon).reshape(tensor_length, 3, 3)
    omega = computeVec3FromSkewSymmMat(omegahat, symm_epsilon).reshape(tensor_length, 3)
    norm_omega = npla.norm(omega, ord=2, axis=1).reshape(tensor_length, 1, 1)
    
    A = (getTensorEye(tensor_length, 3)
         + (np.tile(((1.0-np.cos(norm_omega))/np.square(norm_omega+denom_epsilon)), (1, 3, 3)) * omegahat)
         + (np.tile(((norm_omega-np.sin(norm_omega))/(np.square(norm_omega+denom_epsilon)*(norm_omega+denom_epsilon))), (1, 3, 3)) 
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


if __name__=='__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    diff_epsilon = 1.0e-9
    
    print ""
    print "SO(3) Log Mapping Test:"
    R1 = np.eye(3)
    print "R1 = \n", R1
    start = time.time()
    npLogR1 = scla.logm(R1)
    print "npLogR1 = \n", npLogR1
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    LogR1 = computeRotationMatrixLogMap(R1)
    print "AnalyticalLogR1 = \n", LogR1
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npLogR1 - LogR1)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    theta2 = np.pi/2.0
    R2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(theta2), -np.sin(theta2)],
                   [0.0, np.sin(theta2), np.cos(theta2)]])
    print "R2 = \n", R2
    start = time.time()
    npLogR2 = scla.logm(R2)
    print "npLogR2 = \n", npLogR2
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    LogR2 = computeRotationMatrixLogMap(R2)
    print "AnalyticalLogR2 = \n", LogR2
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npLogR2 - LogR2)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    theta3 = np.random.rand(1)[0]
    R3 = np.array([[np.cos(theta3), 0.0, np.sin(theta3)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(theta3), 0.0, np.cos(theta3)]])
    print "R3 = \n", R3
    start = time.time()
    npLogR3 = scla.logm(R3)
    print "npLogR3 = \n", npLogR3
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    LogR3 = computeRotationMatrixLogMap(R3)
    print "AnalyticalLogR3 = \n", LogR3
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npLogR3 - LogR3)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    print ""
    print "so(3) Exp Mapping Test:"
    omega1 = np.zeros(3,)
    omegahat1 = computeSkewSymmMatFromVec3(omega1)
    print "omega1 = \n", omega1
    print "omegahat1 = \n", omegahat1
    start = time.time()
    npExpomegahat1 = scla.expm(omegahat1)
    print "npExpomegahat1 = \n", npExpomegahat1
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    Expomegahat1 = computeRotationMatrixExpMap(omegahat1)
    print "AnalyticalExpomegahat1 = \n", Expomegahat1
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npExpomegahat1 - Expomegahat1)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    omega2 = np.random.rand(3)
    omegahat2 = computeSkewSymmMatFromVec3(omega2)
    print "omega2 = \n", omega2
    print "omegahat2 = \n", omegahat2
    start = time.time()
    npExpomegahat2 = scla.expm(omegahat2)
    print "npExpomegahat2 = \n", npExpomegahat2
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    Expomegahat2 = computeRotationMatrixExpMap(omegahat2)
    print "AnalyticalExpomegahat2 = \n", Expomegahat2
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npExpomegahat2 - Expomegahat2)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    print ""
    print "SE(3) Log Mapping Test:"
    T1 = np.eye(4)
    T1[:3,:3] = R1
    T1[:3,3] = np.random.rand(3)
    print "T1 = \n", T1
    start = time.time()
    npLogT1 = scla.logm(T1)
    print "npLogT1 = \n", npLogT1
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    LogT1 = computeHomogeneousTransformationMatrixLogMap(T1)
    print "AnalyticalLogT1 = \n", LogT1
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npLogT1 - LogT1)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    T2 = np.eye(4)
    T2[:3,:3] = R2
    T2[:3,3] = np.random.rand(3)
    print "T2 = \n", T2
    start = time.time()
    npLogT2 = scla.logm(T2)
    print "npLogT2 = \n", npLogT2
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    LogT2 = computeHomogeneousTransformationMatrixLogMap(T2)
    print "AnalyticalLogT2 = \n", LogT2
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npLogT2 - LogT2)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    print ""
    print "se(3) Exp Mapping Test:"
    kseehat1 = np.zeros((4,4))
    kseehat1[:3,:3] = omegahat1
    kseehat1[:3,3] = np.random.rand(3)
    print "kseehat1 = \n", kseehat1
    start = time.time()
    npExpkseehat1 = scla.expm(kseehat1)
    print "npExpkseehat1 = \n", npExpkseehat1
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    Expkseehat1 = computeHomogeneousTransformationMatrixExpMap(kseehat1)
    print "AnalyticalExpkseehat1 = \n", Expkseehat1
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npExpkseehat1 - Expkseehat1)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    kseehat2 = np.zeros((4,4))
    kseehat2[:3,:3] = omegahat2
    kseehat2[:3,3] = np.random.rand(3)
    print "kseehat2 = \n", kseehat2
    start = time.time()
    npExpkseehat2 = scla.expm(kseehat2)
    print "npExpkseehat2 = \n", npExpkseehat2
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    Expkseehat2 = computeHomogeneousTransformationMatrixExpMap(kseehat2)
    print "AnalyticalExpkseehat2 = \n", Expkseehat2
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npExpkseehat2 - Expkseehat2)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    print ""
    print "Inverse Homogeneous Transformation Matrix Test:"
    print "T1 = \n", T1
    start = time.time()
    npInvT1 = npla.inv(T1)
    print "npInvT1 = \n", npInvT1
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    InvT1 = computeInverseHomogeneousTransformMatrix(T1)
    print "AnalyticalInvT1 = \n", InvT1
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npInvT1 - InvT1)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    print "T2 = \n", T2
    start = time.time()
    npInvT2 = npla.inv(T2)
    print "npInvT2 = \n", npInvT2
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    InvT2 = computeInverseHomogeneousTransformMatrix(T2)
    print "AnalyticalInvT2 = \n", InvT2
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(npInvT2 - InvT2)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    print ""
    print "SO(3) Cubed/Composition Test:"
    print "R3 = \n", R3
    start = time.time()
    R3cubed = np.matmul(R3, np.matmul(R3, R3))
    print "R3cubed = \n", R3cubed
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    exp_3Xlog_R3 = computeRotationMatrixExpMap(3 * LogR3)
    print "exp_3Xlog_R3 = \n", exp_3Xlog_R3
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(R3cubed - exp_3Xlog_R3)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    print ""
    print "SE(3) Cubed/Composition Test:"
    print "T2 = \n", T2
    start = time.time()
    T2cubed = np.matmul(T2, np.matmul(T2, T2))
    print "T2cubed = \n", T2cubed
    end = time.time()
    print "Computation Time: ", (end - start)
    start = time.time()
    exp_3Xlog_T2 = computeHomogeneousTransformationMatrixExpMap(3 * LogT2)
    print "exp_3Xlog_T2 = \n", exp_3Xlog_T2
    end = time.time()
    print "Computation Time: ", (end - start)
    diff = npla.norm(T2cubed - exp_3Xlog_T2)
    assert diff < diff_epsilon, "diff = %f" % diff
    
    ## The following is NOT the same (incorrect algebra; NOT commutative)
#    print ""
#    print "SO(3) Composition Test:"
#    print "R2 = \n", R2
#    print "R3 = \n", R3
#    start = time.time()
#    R2composeR3 = np.matmul(R2, R3)
#    print "R2composeR3 = \n", R2composeR3
#    end = time.time()
#    print "Computation Time: ", (end - start)
#    start = time.time()
#    exp_logR2_plus_logR3 = computeRotationMatrixExpMap(LogR2 + LogR3)
#    print "exp_logR2_plus_logR3 = \n", exp_logR2_plus_logR3
#    end = time.time()
#    print "Computation Time: ", (end - start)
#    diff = npla.norm(R2composeR3 - exp_logR2_plus_logR3)
#    assert diff < diff_epsilon, "diff = %f" % diff
    
    ## The following is NOT the same (incorrect algebra; NOT commutative)
#    print ""
#    print "SE(3) Composition Test:"
#    print "T1 = \n", T1
#    print "T2 = \n", T2
#    start = time.time()
#    T1composeT2 = np.matmul(T1, T2)
#    print "T1composeT2 = \n", T1composeT2
#    end = time.time()
#    print "Computation Time: ", (end - start)
#    start = time.time()
#    exp_logT1_plus_logT2 = computeHomogeneousTransformationMatrixExpMap(LogT1 + LogT2)
#    print "exp_logT1_plus_logT2 = \n", exp_logT1_plus_logT2
#    end = time.time()
#    print "Computation Time: ", (end - start)
#    diff = npla.norm(T1composeT2 - exp_logT1_plus_logT2)
#    assert diff < diff_epsilon, "diff = %f" % diff

    t = np.array([[1,2,3.0],[4,5,6],[7,8,9],[101,103,105],[201,53,405]])
    Q = np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0],[0.7073883, 0, 0.4998009, 0.4998009]])
    T = computeHomogeneousTransformMatrix(t, Q)
    np.set_printoptions(precision=3,suppress=True)
    print "T = \n", T