/*
 * DMPState.h
 *
 *  Data object that holds variables that can be needed for estimating
 *  the current DMP state including position, velocity, acceleration,
 *  and time
 *
 *  Created on: Dec 12, 2014
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef DMP_STATE_H
#define DMP_STATE_H

#include <stddef.h>
#include <stdlib.h>

#include <iostream>
#include <limits>

#include "dmp/utility/DefinitionsBase.h"
#include "dmp/utility/RealTimeAssertor.h"

namespace dmp {

class DMPState {
 protected:
  uint dmp_num_dimensions;  // number of dimensions in the DMP, for example:
                            // Cartesian DMP has 3 dimensions, Quaternion DMP
                            // has 4 dimensions, etc.
  double time;  // "time" variable here is used to accomodate trajectory
                // training data which came from a different sampling rate

  // We use VectorN (instead of VectorNPtr) here for x, xd, and xdd,
  // because the overall DMPState data structure MUST BE real-time-safe:
  VectorN x;   // vector of length N, with n-th component represent the position
               // of the n-th DMP trajectory
  VectorN xd;  // vector of length N, with n-th component represent the velocity
               // of the n-th DMP trajectory
  VectorN xdd;  // vector of length N, with n-th component represent the
                // acceleration of the n-th DMP trajectory
  // double            sensitive_unused_variable;	// toogle
  // commenting/uncommenting this variable if DMP does NOT work (very STRANGE
  // problem, unknown reason)

  // The following field(s) are written as RAW pointers because we just want to
  // point to other data structure(s) that might outlive the instance of this
  // class:
  RealTimeAssertor* rt_assertor;

 public:
  DMPState();

  /**
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DMPState(RealTimeAssertor* real_time_assertor);

  /**
   * @param dmp_num_dimensions_init Initialization of the number of dimensions
   * that will be used in the DMP
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DMPState(uint dmp_num_dimensions_init, RealTimeAssertor* real_time_assertor);

  /**
   * @param x_init Initialization value of N-dimensional position vector
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DMPState(const VectorN& x_init, RealTimeAssertor* real_time_assertor);

  /**
   * @param x_init Initialization value of N-dimensional position vector
   * @param xd_init Initialization value of N-dimensional velocity vector
   * @param xdd_init Initialization value of N-dimensional acceleration vector
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DMPState(const VectorN& x_init, const VectorN& xd_init,
           const VectorN& xdd_init, RealTimeAssertor* real_time_assertor);

  /**
   * @param x_init Initialization value of N-dimensional position vector
   * @param xd_init Initialization value of N-dimensional velocity vector
   * @param xdd_init Initialization value of N-dimensional acceleration vector
   * @param time_init Time instant where this state exists within the course of
   * the trajectory
   * @param real_time_assertor Real-Time Assertor for troubleshooting and
   * debugging
   */
  DMPState(const VectorN& x_init, const VectorN& xd_init,
           const VectorN& xdd_init, double time_init,
           RealTimeAssertor* real_time_assertor);

  /**
   * @param other_dmpstate Another DMPState whose value will be copied into this
   * DMPState
   */
  DMPState(const DMPState& other_dmpstate);

  /**
   * Checks whether the dimensions of x, xd, and xdd are all equivalent to
   * dmp_num_dimensions and time is greater than or equal to 0.0.
   *
   * @return DMPState is valid (true) or DMPState is invalid (false)
   */
  virtual bool isValid() const;

  /**
   * Assignment operator ( operator= ) overloading.
   *
   * @param other_dmpstate Another DMPState whose value will be copied into this
   * DMPState
   */
  virtual DMPState& operator=(const DMPState& other_dmpstate);

  virtual uint getDMPNumDimensions() const;
  virtual VectorN getX() const;
  virtual VectorN getXd() const;
  virtual VectorN getXdd() const;
  virtual double getTime() const;

  virtual bool setX(const VectorN& new_x);
  virtual bool setXd(const VectorN& new_xd);
  virtual bool setXdd(const VectorN& new_xdd);
  virtual bool setTime(double new_time);

  ~DMPState();
};

}  // namespace dmp
#endif
