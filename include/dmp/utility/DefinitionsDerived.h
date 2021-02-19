/*
 * DefinitionsDerived.h
 *
 *  Defining general derived types (separated from DefinitionsBase.h to avoid circular dependencies).
 *
 *  Created on: Oct 23, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef DEFINITIONS_DERIVED_H
#define DEFINITIONS_DERIVED_H

#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_state/QuaternionDMPState.h"
#include <vector>


namespace dmp
{

#define MAX_TRAJ_SIZE               (20 * MAX_TASK_SERVO_RATE)  // maximum trajectory size is about 20 seconds (for robots with MAX_TASK_SERVO_RATE, longer for robots with lower task servo rate) of demonstrated trajectory
#define MAX_DMP_STATE_SIZE          (1 + (3 * MAX_DMP_NUM_DIMENSIONS))  // maximum DMP state size is 1 (for time) + 3 (for position, velocity, and acceleration) times MAX_DMP_NUM_DIMENSIONS
#define MAX_SAVE_BUFFER_COL_SIZE    MAX(MAX_DMP_STATE_SIZE, (MAX_MODEL_SIZE+1))
#define MAX_NUM_POINTS              100

    typedef std::vector< Vector3 >                          Points;
    typedef std::shared_ptr< Points >                       PointsPtr;

    typedef std::vector< Points >                           PointsVector;
    typedef std::shared_ptr< PointsVector >                 PointsVectorPtr;

    typedef std::vector< dmp::DMPState >                    DMPStates;

    typedef std::shared_ptr< dmp::DMPState >                DMPStatePtr;

    typedef std::vector< dmp::DMPState >                    Trajectory;
    typedef std::shared_ptr< Trajectory >                   TrajectoryPtr;

    typedef std::vector< dmp::QuaternionDMPState >          QuaternionTrajectory;
    typedef std::shared_ptr< QuaternionTrajectory >         QuaternionTrajectoryPtr;

    typedef std::vector< Trajectory >                       TrajectorySet;
    typedef std::shared_ptr< TrajectorySet >                TrajectorySetPtr;

    typedef std::vector< QuaternionTrajectory >             QuaternionTrajectorySet;
    typedef std::shared_ptr< QuaternionTrajectorySet >      QuaternionTrajectorySetPtr;

    typedef std::vector< TrajectorySet >                    DemonstrationGroupSet;
    typedef std::shared_ptr< DemonstrationGroupSet >        DemonstrationGroupSetPtr;

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, MAX_SAVE_BUFFER_COL_SIZE>    MatrixTxSBC;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, (MAX_MODEL_SIZE+1)>          MatrixTxMp1;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, MAX_DMP_STATE_SIZE>          MatrixTxS;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, (MAX_DMP_NUM_DIMENSIONS+1)>  MatrixTxNp1;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::AutoAlign, MAX_TRAJ_SIZE>                                           MatrixTx4;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::AutoAlign, MAX_TRAJ_SIZE>                                           MatrixTx3;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::AutoAlign, MAX_TRAJ_SIZE>                                           MatrixTx2;
    typedef Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::AutoAlign, 4, MAX_TRAJ_SIZE>                                        Matrix4xT;
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_TRAJ_SIZE>                                        Matrix3xT;
    typedef Eigen::Matrix<double, 4, Eigen::Dynamic, Eigen::AutoAlign, 4, MAX_NUM_POINTS>                                       Matrix4xP;
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_NUM_POINTS>                                       Matrix3xP;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_TRAJ_SIZE>                                           VectorT;

    typedef std::shared_ptr< MatrixTxSBC >    MatrixTxSBCPtr;
    typedef std::shared_ptr< MatrixTxMp1 >    MatrixTxMp1Ptr;
    typedef std::shared_ptr< MatrixTxS >      MatrixTxSPtr;
    typedef std::shared_ptr< MatrixTxNp1 >    MatrixTxNp1Ptr;
    typedef std::shared_ptr< MatrixTx4 >      MatrixTx4Ptr;
    typedef std::shared_ptr< MatrixTx3 >      MatrixTx3Ptr;
    typedef std::shared_ptr< MatrixTx2 >      MatrixTx2Ptr;
    typedef std::shared_ptr< Matrix4xT >      Matrix4xTPtr;
    typedef std::shared_ptr< Matrix3xT >      Matrix3xTPtr;
    typedef std::shared_ptr< VectorT >        VectorTPtr;

#define ZeroMatrixTxSBC(T,SBC)  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, MAX_SAVE_BUFFER_COL_SIZE>::Zero(T,SBC)
#define ZeroMatrixTxMp1(T,Mp1)  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, (MAX_MODEL_SIZE+1)>::Zero(T,Mp1)
#define ZeroMatrixTxS(T,S)      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, MAX_DMP_STATE_SIZE>::Zero(T,S)
#define ZeroMatrixTxNp1(T,Np1)  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, (MAX_DMP_NUM_DIMENSIONS+1)>::Zero(T,Np1)
#define ZeroMatrixTx4(T)        Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::AutoAlign, MAX_TRAJ_SIZE>::Zero(T,4)
#define ZeroMatrixTx3(T)        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::AutoAlign, MAX_TRAJ_SIZE>::Zero(T,3)
#define ZeroMatrixTx2(T)        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::AutoAlign, MAX_TRAJ_SIZE>::Zero(T,2)
#define ZeroMatrix3xT(T)        Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_TRAJ_SIZE>::Zero(3,T)

}

#endif
