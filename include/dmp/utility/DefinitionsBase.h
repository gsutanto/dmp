/*
 * DefinitionsBase.h
 *
 *  Defining general base types and limits of parameters
 *  ('general' means being used across many different places within this DMP package).
 *  Limits of parameters, such as maximum DMP dimensions, maximum task servo rate, maximum trajectory size, 
 *  maximum DMP basis functions model size, etc. are important in determining the size of some data structures.
 *
 *  Created on: Aug 28, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef DEFINITIONS_BASE_H
#define DEFINITIONS_BASE_H

#include <Eigen/Core>
#include <memory>
#include <filesystem>


namespace dmp
{

#define MIN(a,b)                    (((a)<(b))?(a):(b))
#define MAX(a,b)                    (((a)>(b))?(a):(b))
#define MAX_TASK_SERVO_RATE         1260    // assuming no robot running this DMP package has task servo rate GREATER THAN <this value> Hertz
#define MAX_DMP_NUM_DIMENSIONS      20      // dmp_num_dimensions is 1 for DMPDiscrete1D, and 3 for CartesianCoordDMP
#define MAX_MODEL_SIZE              50     // maximum DMP basis functions model size
#define MIN_TAU                     0.01    // minimum value of tau
#define MAX_TAU_LEARNING_UNROLL     10      // maximum value of tau for unrolling DMP in a learning process; e.g. this is used in TransformCouplingLearnObsAvoid.cpp (10.0 seconds)
#define MAX_NUM_UNROLL_TRAJ_STATES  (MAX_TAU_LEARNING_UNROLL*MAX_TASK_SERVO_RATE)

enum DMPFormulation
{
    _SCHAAL_DMP_                            = 0,
    _HOFFMANN_DMP_                          = 1
};

enum DMPForcingTermLearningMethod
{
    _SCHAAL_LWR_METHOD_                     = 0,
    _JPETERS_GLOBAL_LEAST_SQUARES_METHOD_   = 1
};

typedef std::vector< double >               DoubleVector;
typedef std::shared_ptr< DoubleVector >     DoubleVectorPtr;

typedef std::shared_array< char >           CharArr;

typedef Eigen::MatrixXd                                                                                                     MatrixXxX;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_DMP_NUM_DIMENSIONS, MAX_MODEL_SIZE>     MatrixNxM;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_MODEL_SIZE, MAX_MODEL_SIZE>             MatrixMxM;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_MODEL_SIZE>                                       Matrix3xM;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_NUM_UNROLL_TRAJ_STATES>                           Matrix3xU;
typedef Eigen::Matrix4d                                                                                                     Matrix4x4;
typedef Eigen::Matrix3d                                                                                                     Matrix3x3;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_DMP_NUM_DIMENSIONS>                                  VectorN;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_MODEL_SIZE>                                          VectorM;
typedef Eigen::Vector3d                                                                                                     Vector3;
typedef Eigen::Vector4d                                                                                                     Vector4;

typedef std::shared_ptr< MatrixXxX >  MatrixXxXPtr;
typedef std::shared_ptr< MatrixNxM >  MatrixNxMPtr;
typedef std::shared_ptr< Matrix3xM >  Matrix3xMPtr;
typedef std::shared_ptr< Matrix3xU >  Matrix3xUPtr;
typedef std::shared_ptr< VectorN >    VectorNPtr;
typedef std::shared_ptr< VectorM >    VectorMPtr;

#define ZeroMatrixNxM(N,M)      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_DMP_NUM_DIMENSIONS, MAX_MODEL_SIZE>::Zero(N,M)
#define ZeroMatrix3xM(M)        Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_MODEL_SIZE>::Zero(3,M)
#define ZeroMatrix3xU(U)        Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::AutoAlign, 3, MAX_NUM_UNROLL_TRAJ_STATES>::Zero(3,U)
#define ZeroMatrix4x4           Eigen::Matrix4d::Zero()
#define ZeroMatrix3x3           Eigen::Matrix3d::Zero()
#define EyeMatrixXxX(X1,X2)     Eigen::MatrixXd::Identity(X1,X2)
#define EyeMatrixNxN(N)         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MAX_DMP_NUM_DIMENSIONS, MAX_DMP_NUM_DIMENSIONS>::Identity(N,N)
#define EyeMatrix3x3            Eigen::Matrix3d::Identity()
#define EyeMatrix4x4            Eigen::Matrix4d::Identity()
#define ZeroVectorN(N)          Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_DMP_NUM_DIMENSIONS>::Zero(N)
#define ZeroVectorM(M)          Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_MODEL_SIZE>::Zero(M)
#define ZeroVector3             Eigen::Vector3d::Zero()
#define ZeroVector4             Eigen::Vector4d::Zero()
#define RandomVector3           Eigen::Vector3d::Zero().unaryExpr(std::ptr_fun(sample_normal_distribution))
#define ZeroRowVector3          Eigen::Vector3d::Zero().transpose()
#define OnesVectorN(N)          Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_DMP_NUM_DIMENSIONS>::Ones(N)
#define OnesVectorM(M)          Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, MAX_MODEL_SIZE>::Ones(M)

struct DMPDiscreteParams
{
    MatrixNxM   weights;
    VectorN     A_learn;
};

typedef std::vector< DMPDiscreteParams >            DMPDiscreteParamsSet;
typedef std::shared_ptr< DMPDiscreteParamsSet >     DMPDiscreteParamsSetPtr;

/*static class {
public:
    template<typename T>
    operator std::shared_ptr<T>() { return std::shared_ptr<T>(); }
} nullPtr;*/

}

#endif
