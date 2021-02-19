/*
 * DefinitionsLearnObsAvoid.h
 *
 *  Defining types and limits of parameters for obstacle-avoidance-by-learning/demonstration (LOA) coupling term.
 *
 *  Created on: Dec 12, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef DEFINITIONS_LEARN_OBS_AVOID_H
#define DEFINITIONS_LEARN_OBS_AVOID_H

#include "dmp/utility/DefinitionsBase.h"
#include "dmp/utility/DefinitionsDerived.h"

namespace dmp
{

#define MAX_LOA_GRID_SIZE                               10  // maximum grid size (for beta and k parameters)
#define MAX_AF_H14_NUM_INDEP_FEAT_OBS_POINTS            10  // maximum number of obstacle points that will be considered/evaluated as separate/independent feature sets for approximating the obstacle(s), for AF_H14 feature model
#define MAX_AF_H14_NUM_FEATURE_MATRIX_ROWS              (MAX_LOA_GRID_SIZE*((MAX_AF_H14_NUM_INDEP_FEAT_OBS_POINTS*MAX_LOA_GRID_SIZE)+1))    // maximum length of the AF_H14 feature vector of EACH Cartesian axis for obstacle avoidance = ((#O * G^2) + G) => (for each of Cartesian axis: x,y,z)
#define MAX_PF_DYN2_NUM_FEATURE_MATRIX_ROWS             (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)   // maximum length of the PF_DYN2 feature vector of EACH Cartesian axis for obstacle avoidance = (G^2) => (for each of Cartesian axis: x,y,z)
#define MAX_KGF_NUM_FEATURE_MATRIX_ROWS                 (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)
#define MAX_NN_NUM_FEATURE_VECTOR_ROWS                  200
#define MAX_NN_NUM_INPUT_NODES                          MAX_NN_NUM_FEATURE_VECTOR_ROWS
#define MAX_NN_NUM_HIDDEN_NODES                         50
#define MAX_NN_NUM_OUTPUT_NODES                         3
#define MAX_LOA_NUM_FEATURE_MATRIX_ROWS                 (MAX_AF_H14_NUM_FEATURE_MATRIX_ROWS+MAX_PF_DYN2_NUM_FEATURE_MATRIX_ROWS+MAX_KGF_NUM_FEATURE_MATRIX_ROWS)    // maximum length of overall feature vector of EACH Cartesian axis (x,y,z) for obstacle avoidance
#define MAX_LOA_NUM_FEATURE_VECTOR_SIZE                 (3*MAX_LOA_NUM_FEATURE_MATRIX_ROWS) // maximum length (total rows) of the feature vector of ALL 3 Cartesian axes (x,y,z) for obstacle avoidance
#define MIN_ROTATION_AXIS_NORM                          0.0001
#define MIN_POINT_OBSTACLE_VECTOR_PREALLOCATION         200
#define DEFAULT_KERNEL_SPREAD                           1.0

enum DMPCouplingTermLearnObsAvoidDistanceKernelScaleGridMode
{
    _LINEAR_                                        = 1,
    _QUADRATIC_                                     = 2,
    _INVERSE_QUADRATIC_                             = 3,
    _NUM_DISTANCE_KERNEL_SCALE_GRID_MODES_          = 3
};

enum DMPCouplingTermLearnObsAvoidAFH14FeatureMethod
{
    _AF_H14_NO_USE_                                 = 0,
    _AF_H14_                                        = 1,
    _AF_H14_SUM_OBS_POINTS_CONTRIBUTION_            = 2,
    _AF_H14_NO_PHI3_                                = 3,
    _AF_H14_NO_PHI3_SUM_OBS_POINTS_CONTRIBUTION_    = 4,
    _NUM_AF_H14_FEATURE_METHODS_                    = 4
};

enum DMPCouplingTermLearnObsAvoidPFDYN2FeatureMethod
{
    _PF_DYN2_NO_USE_                                = 0,
    _PF_DYN2_                                       = 1,
    _NUM_PF_DYN2_FEATURE_METHODS_                   = 1
};

enum DMPCouplingTermLearnObsAvoidKGFFeatureMethod
{
    _KGF_NO_USE_                                    = 0,
    _KGF_                                           = 1,
    _NUM_KGF_FEATURE_METHODS_                       = 1
};

enum DMPCouplingTermLearnObsAvoidNeuralNetworkFeatureMethod
{
    _NN_NO_USE_                                     = 0,
    _NN_                                            = 1,
    _NUM_NN_FEATURE_METHODS_                        = 1
};

enum DMPCouplingTermLearnObsAvoidDataFileFormat
{
    _SPHERE_OBSTACLE_CENTER_RADIUS_FILE_FORMAT_                         = 1,
    _VICON_MARKERS_DATA_FILE_FORMAT_                                    = 2,
    _NUM_DATA_FILE_FORMATS_                                             = 2
};

typedef std::vector< dmp::DMPState >                ObstacleStates;
typedef boost::shared_ptr< ObstacleStates >         ObstacleStatesPtr;

typedef std::vector< ObstacleStates >               ObstacleStatesVector;
typedef boost::shared_ptr< ObstacleStatesVector >   ObstacleStatesVectorPtr;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>                 MatrixloaTxF3;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, (3*MAX_TRAJ_SIZE), MAX_LOA_NUM_FEATURE_MATRIX_ROWS>             Matrixloa3TxF1;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::AutoAlign, MAX_LOA_NUM_FEATURE_MATRIX_ROWS>                                             MatrixloaF1x3;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::AutoAlign, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>                                             MatrixloaF3x3;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_LOA_GRID_SIZE, MAX_LOA_GRID_SIZE>                           MatrixloaGxG;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE), MAX_LOA_GRID_SIZE>       MatrixloaG2xG;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_NUM_UNROLL_TRAJ_STATES, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>    MatrixloaUxF3;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_NN_NUM_HIDDEN_NODES, MAX_NN_NUM_INPUT_NODES>                MatrixloaNN_HxI;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_NN_NUM_HIDDEN_NODES, MAX_NN_NUM_HIDDEN_NODES>               MatrixloaNN_HxH;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_NN_NUM_OUTPUT_NODES, MAX_NN_NUM_HIDDEN_NODES>               MatrixloaNN_OxH;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>                                          Matrixloa3xF3;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_AF_H14_NUM_INDEP_FEAT_OBS_POINTS>                                     Matrixloa3xO;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_LOA_GRID_SIZE>                                                        Matrixloa3xG;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)>                                    Matrixloa3xG2;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)>                  Matrixloa3xG3;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, (3*MAX_TRAJ_SIZE)>                                                           Vectorloa3T;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_LOA_GRID_SIZE>                                                           VectorloaG;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)>                                       VectorloaG2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)>                     VectorloaG3;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_LOA_NUM_FEATURE_MATRIX_ROWS>                                             VectorloaF1;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>                                             VectorloaF3;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_NN_NUM_FEATURE_VECTOR_ROWS>                                              VectorloaNN_I;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_NN_NUM_HIDDEN_NODES>                                                     VectorloaNN_H;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_NN_NUM_OUTPUT_NODES>                                                     VectorloaNN_O;

typedef boost::shared_ptr< MatrixloaTxF3 >      MatrixloaTxF3Ptr;
typedef boost::shared_ptr< Matrixloa3TxF1 >     Matrixloa3TxF1Ptr;
typedef boost::shared_ptr< MatrixloaF1x3 >      MatrixloaF1x3Ptr;
typedef boost::shared_ptr< MatrixloaGxG >       MatrixloaGxGPtr;
typedef boost::shared_ptr< MatrixloaG2xG >      MatrixloaG2xGPtr;
typedef boost::shared_ptr< MatrixloaF3x3 >      MatrixloaF3x3Ptr;
typedef boost::shared_ptr< MatrixloaUxF3 >      MatrixloaUxF3Ptr;
typedef boost::shared_ptr< MatrixloaNN_HxI >    MatrixloaNN_HxIPtr;
typedef boost::shared_ptr< MatrixloaNN_HxH >    MatrixloaNN_HxHPtr;
typedef boost::shared_ptr< MatrixloaNN_OxH >    MatrixloaNN_OxHPtr;
typedef boost::shared_ptr< Matrixloa3xF3 >      Matrixloa3xF3Ptr;
typedef boost::shared_ptr< Matrixloa3xO >       Matrixloa3xOPtr;
typedef boost::shared_ptr< Matrixloa3xG >       Matrixloa3xGPtr;
typedef boost::shared_ptr< Matrixloa3xG2 >      Matrixloa3xG2Ptr;
typedef boost::shared_ptr< Matrixloa3xG3 >      Matrixloa3xG3Ptr;
typedef boost::shared_ptr< Vectorloa3T >        Vectorloa3TPtr;
typedef boost::shared_ptr< VectorloaG >         VectorloaGPtr;
typedef boost::shared_ptr< VectorloaG2 >        VectorloaG2Ptr;
typedef boost::shared_ptr< VectorloaG3 >        VectorloaG3Ptr;
typedef boost::shared_ptr< VectorloaF1 >        VectorloaF1Ptr;
typedef boost::shared_ptr< VectorloaF3 >        VectorloaF3Ptr;
typedef boost::shared_ptr< VectorloaNN_I >      VectorloaNN_IPtr;
typedef boost::shared_ptr< VectorloaNN_H >      VectorloaNN_HPtr;
typedef boost::shared_ptr< VectorloaNN_O >      VectorloaNN_OPtr;

#define ZeroMatrixloaTxF3(T,F3) Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_TRAJ_SIZE, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>::Zero(T,F3)
#define ZeroMatrixloaF1x3(F1)   Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::AutoAlign, MAX_LOA_NUM_FEATURE_MATRIX_ROWS>::Zero(F1,3)
#define ZeroMatrixloaF3x3(F3)   Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::AutoAlign, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>::Zero(F3,2)
#define ZeroMatrixloaUxF3(U,F3) Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_NUM_UNROLL_TRAJ_STATES, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>::Zero(U,F3)
#define ZeroMatrixloa3xG(G)     Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_LOA_GRID_SIZE>::Zero(3,G)
#define ZeroMatrixloa3xG2(G2)   Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)>::Zero(3,G2)
#define ZeroVectorloaF3(F3)     Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_LOA_NUM_FEATURE_VECTOR_SIZE>::Zero(F3,1)
#define ZeroVectorloaNN_I(M)    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_NN_NUM_FEATURE_VECTOR_ROWS>::Zero(M,1)
#define OnesVectorloaG2(G2)     Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)>::Ones(G2,1)
#define OnesVectorloaG3(G3)     Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, (MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE*MAX_LOA_GRID_SIZE)>::Ones(G3,1)
#define OnesVectorloaNN_I(I)    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_NN_NUM_FEATURE_VECTOR_ROWS>::Ones(I,1)
#define OnesVectorloaNN_O(O)    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_NN_NUM_OUTPUT_NODES>::Ones(O,1)

}

#endif
