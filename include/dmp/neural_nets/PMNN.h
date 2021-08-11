/*
 * PMNN.h
 *
 *  Class for Phase-Modulated Neural Network (PMNN:
 *  feedforward neural network with phase modulation on the final hidden layer).
 *  Each dimension has its own network.
 *
 *  Created on: June 24, 2017
 *  Author: Giovanni Sutanto
 */

#ifndef PMNN_H
#define PMNN_H

#include "dmp/utility/DataIO.h"
#include "dmp/utility/DefinitionsDerived.h"
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/utility/utility.h"

namespace dmp {

#define MAX_NN_NUM_NODES_PER_LAYER 100

enum NNActivationFunction { _IDENTITY_ = 0, _TANH_ = 1, _RELU_ = 2 };

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign,
                      MAX_NN_NUM_NODES_PER_LAYER, MAX_NN_NUM_NODES_PER_LAYER>
    MatrixNN_NxN;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign,
                      MAX_NN_NUM_NODES_PER_LAYER>
    VectorNN_N;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign,
                      2, MAX_NN_NUM_NODES_PER_LAYER>
    MatrixNN_2xN;

typedef std::unique_ptr<MatrixNN_NxN> MatrixNN_NxNPtr;
typedef std::unique_ptr<VectorNN_N> VectorNN_NPtr;
typedef std::unique_ptr<MatrixNN_2xN> MatrixNN_2xNPtr;

#define ZeroVectorNN_N(N)                                    \
  Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::AutoAlign, \
                MAX_NN_NUM_NODES_PER_LAYER>::Zero(N)

class PMNN {
 protected:
  uint num_dimensions;  // number of dimensions; this is equal 3 for Cartesian
                        // Coordinate DMP, and also equal 3 for Quaternion DMP

  std::vector<std::vector<MatrixNN_NxNPtr> > weights;
  std::vector<std::vector<MatrixNN_2xNPtr> > biases;

  std::vector<uint> topology;

  std::vector<uint> activation_functions;  // activation functions corresponding
                                           // to the specified topology

  DataIO data_io;

  RealTimeAssertor* rt_assertor;

 public:
  PMNN();

  /**
   * @param num_dimensions_init Initialization of the number of dimensions that
   * will be used
   * @param topology_init Initialization of the neural network topology that
   * will be used
   * @param activation_functions_init Initialization of the neural network
   * topology's activation functions that will be used
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  PMNN(uint num_dimensions_init, std::vector<uint> topology_init,
       std::vector<uint> activation_functions_init,
       RealTimeAssertor* real_time_assertor);

  /**
   * Checks whether this special neural network is valid or not.
   *
   * @return Valid (true) or invalid (false)
   */
  bool isValid();

  /**
   * NOT REAL-TIME!!!
   *
   * Load neural network parameters from text files in the specified directory.
   *
   * @param dir_path Directory containing the parameters
   * @param start_idx Start sub-directory index to read from
   * @return Success or failure
   */
  bool loadParams(const char* dir_path, uint start_idx);

  /**
   * MUST BE REAL-TIME!!!
   *
   * Compute prediction, based on given input and phase kernel modulator.
   *
   * @param input Input vector
   * @param phase_kernel_modulation (Input) vector of phase kernel modulation
   * @param output Output vector
   * @return Success or failure
   */
  bool computePrediction(const VectorNN_N& input,
                         const VectorNN_N& phase_kernel_modulation,
                         VectorNN_N& output, int start_index = 0,
                         int end_index = -1);

  ~PMNN();
};
}  // namespace dmp
#endif
