/*
 * utility.h
 * (NON-Real-Time!!!)
 *
 *  Compilation of utility functions for this DMP package, such as
 *  information extraction from file(s), linear algebra computations, etc.
 *
 *  Created on: September 05, 2015
 *  Author: Yevgen Chebotar and Giovanni Sutanto
 */

#ifndef UTILITY_H
#define UTILITY_H

#include <dirent.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/SVD>
#include <fstream>
#include <string>
#include <vector>

#include "dmp/utility/DefinitionsBase.h"

/*! defines */
enum FileType {
  _DIR_ = 1,  // directory
  _REG_ = 2   // regular file
};

namespace dmp {

/**
 * Sorry for the mess here, C++ compiler did NOT allow me to write
 * the following function definition in utility.cpp file
 * (if I do, I have to instantiate it, which is annoying).
 * Checks whether an Eigen data structure contains a NaN (Not-a-Number) element
 * or not.
 *
 * @param input_mat Matrix whose elements are to be checked as being a NaN or
 * not
 * @return True if input_mat contains a NaN component, false if not
 */
template <class T>
bool containsNaN(const T& input_mat) {
  for (uint j = 0; j < input_mat.cols(); ++j) {
    for (uint i = 0; i < input_mat.rows(); ++i) {
      if (std::isnan(input_mat(i, j))) {
        return true;
      }
    }
  }
  return false;
}

/**
 * NON-REAL-TIME!!!\n
 * Sorry for the mess here, C++ compiler did NOT allow me to write
 * the following function definition in utility.cpp file
 * (if I do, I have to instantiate it, which is annoying).
 * Computes the Moore–Penrose pseudo-inverse of a matrix.
 *
 * @param input_mat Matrix to compute inverse of
 * @return Success or failure
 */
template <class T>
bool pseudoInverse(const T& input_mat, T& output_mat) {
  double epsilon = std::numeric_limits<double>::epsilon();
  Eigen::JacobiSVD<T> svd(input_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = epsilon * std::max(input_mat.cols(), input_mat.rows()) *
                     svd.singularValues().array().abs()(0);

  output_mat = svd.matrixV() *
               (svd.singularValues().array().abs() > tolerance)
                   .select(svd.singularValues().array().inverse(), 0)
                   .matrix()
                   .asDiagonal() *
               svd.matrixU().adjoint();
  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * Telling whether an object, located at object_path is a directory, a file, or
 * something else.
 *
 * @param object_path Path specifying the object's location in the file system
 * @return _DIR_ = 1 (directory), _REG_ = 2 (regular file), 0 (something else),
 * -1 (error)
 */
int file_type(const char* object_path);

/**
 * NON-REAL-TIME!!!\n
 * Counting the total number of incrementally numbered directories inside/under
 * a directory. This function assumes all numbered directories are incremental
 * (with +1 increments), \n starting from 1.
 *
 * @param dir_path Directory whose numbered directories content is to be counted
 * @return The total number of incrementally numbered directories inside/under
 * dir_path
 */
uint countNumberedDirectoriesUnderDirectory(const char* dir_path);

/**
 * NON-REAL-TIME!!!\n
 * Counting the total number of incrementally numbered (text) files inside/under
 * a directory. This function assumes all numbered files are incremental (with
 * +1 increments), \n starting from 1.
 *
 * @param dir_path Directory whose numbered files content is to be counted
 * @return The total number of incrementally numbered files inside/under
 * dir_path
 */
uint countNumberedFilesUnderDirectory(const char* dir_path);

/**
 * Create the specified directory (dir_path), if it is currently not exist yet.
 *
 * @param dir_path The specified directory
 * @return Success or failure
 */
bool createDirIfNotExistYet(const char* dir_path);

/**
 * Compute closest point on a sphere's surface, from an evaluation point.
 *
 * @param evaluation_point The evaluation point/coordinate
 * @param sphere_center_point The sphere's center point/coordinate
 * @param sphere_radius The sphere's radius
 * @return The computed closest point/coordinate on the sphere's surface
 */
Vector3 computeClosestPointOnSphereSurface(const Vector3& evaluation_point,
                                           const Vector3& sphere_center_point,
                                           const double& sphere_radius);

/**
 * NON-REAL-TIME!!!\n
 * Sorry for the mess here, C++ compiler did NOT allow me to write
 * the following function definition in utility.cpp file
 * (if I do, I have to instantiate it, which is annoying).\n
 * Check whether memory pointed by the smart pointer is
 * successfully allocated properly or not.
 * Special for Eigen data structures' smart pointers.
 *
 * @param smart_pointer The smart pointer, whose memory is assumed to be
 * allocated on the heap
 * @param row_size The total number of rows of the allocated Eigen matrix which
 * are instantiated
 * @param col_size The total number of columns of the allocated Eigen matrix
 * which are instantiated
 * @return Success or failure
 */
template <class T>
bool isMemoryAllocated(const std::unique_ptr<T>& smart_pointer, uint row_size,
                       uint col_size) {
  if (smart_pointer == NULL) {
    return false;
  }
  if ((smart_pointer->rows() != row_size) ||
      (smart_pointer->cols() != col_size)) {
    return false;
  }
  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * Sorry for the mess here, C++ compiler did NOT allow me to write
 * the following function definition in utility.cpp file
 * (if I do, I have to instantiate it, which is annoying).
 * IF this function is called at non-real-time mode, it will allocate a memory
 * on the heap for the smart pointer. If called from real-time mode, this
 * function will return failure. Special for Eigen data structures' smart
 * pointers.
 *
 * @param is_real_time Flag indicating whether the function calling this
 * function is real-time or not
 * @param smart_pointer The smart pointer, whose memory will be allocated on the
 * heap
 * @param row_size The total number of rows of the allocated Eigen matrix which
 * are instantiated
 * @param col_size The total number of columns of the allocated Eigen matrix
 * which are instantiated
 * @return Success or failure
 */
template <class T>
bool allocateMemoryIfNonRealTime(bool is_real_time,
                                 std::unique_ptr<T>& smart_pointer,
                                 uint row_size, uint col_size) {
  // input checking:
  // (we are only allowed to allocate memory on heap if we operate in
  // NON-real-time mode!!!)
  if (is_real_time == true) {
    return false;
  }
  if ((row_size < 1) || (col_size < 1)) {
    return false;
  }
  if (T::MaxRowsAtCompileTime !=
      (-1))  // if max #rows of matrix T is PRE-DEFINED:
  {
    if (row_size > T::MaxRowsAtCompileTime) {
      return false;
    }
  }
  if (T::MaxColsAtCompileTime !=
      (-1))  // if max #cols of matrix T is PRE-DEFINED:
  {
    if (col_size > T::MaxColsAtCompileTime) {
      return false;
    }
  }

  // execute memory allocation on the heap:
  smart_pointer.reset(new T(row_size, col_size));

  // check if the memory allocation on the heap is successful or not:
  if (isMemoryAllocated(smart_pointer, row_size, col_size) == false) {
    return false;
  }

  // initialize the content of the allocated memory with zeros:
  *smart_pointer = T::Zero(row_size, col_size);

  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * Sorry for the mess here, C++ compiler did NOT allow me to write
 * the following function definition in utility.cpp file
 * (if I do, I have to instantiate it, which is annoying).
 * Resize and reset (re-zero) the pointed data structure.
 * Special for Eigen data structures' smart pointers.
 *
 * @param smart_pointer The smart pointer, whose memory is allocated on the heap
 * @param row_size New total number of rows
 * @param col_size New total number of columns
 * @return Success or failure
 */
template <class T>
bool resizeAndReset(std::unique_ptr<T>& smart_pointer, uint row_size,
                    uint col_size) {
  // input checking:
  if (smart_pointer == NULL) {
    return false;
  }
  if ((row_size < 1) || (col_size < 1)) {
    return false;
  }
  if (T::MaxRowsAtCompileTime !=
      (-1))  // if max #rows of matrix T is PRE-DEFINED:
  {
    if (row_size > T::MaxRowsAtCompileTime) {
      return false;
    }
  }
  if (T::MaxColsAtCompileTime !=
      (-1))  // if max #cols of matrix T is PRE-DEFINED:
  {
    if (col_size > T::MaxColsAtCompileTime) {
      return false;
    }
  }

  // resize:
  smart_pointer->resize(row_size, col_size);

  // re-zero:
  *smart_pointer = T::Zero(row_size, col_size);

  return (isMemoryAllocated(smart_pointer, row_size, col_size));
}

/**
 * NON-REAL-TIME!!!\n
 * Sorry for the mess here, C++ compiler did NOT allow me to write
 * the following function definition in utility.cpp file
 * (if I do, I have to instantiate it, which is annoying).\n
 * Check whether memory pointed by the smart pointer is
 * successfully allocated properly or not.
 * Special for std::vector data structure's smart pointers.
 *
 * @param smart_pointer The smart pointer, whose memory is assumed to be
 * allocated on the heap
 * @param size The total number of elements of the allocated std::vector data
 * structure which are instantiated
 * @return Success or failure
 */
template <class T>
bool isMemoryAllocated(const std::unique_ptr<T>& smart_pointer, uint size) {
  if (smart_pointer == NULL) {
    return false;
  }
  if (smart_pointer->size() != size) {
    return false;
  }
  return true;
}

/**
 * NON-REAL-TIME!!!\n
 * Sorry for the mess here, C++ compiler did NOT allow me to write
 * the following function definition in utility.cpp file
 * (if I do, I have to instantiate it, which is annoying).
 * IF this function is called at non-real-time mode, it will allocate a memory
 * on the heap for the smart pointer. If called from real-time mode, this
 * function will return failure. Special for std::vector data structure's smart
 * pointers.
 *
 * @param is_real_time Flag indicating whether the function calling this
 * function is real-time or not
 * @param smart_pointer The smart pointer, whose memory will be allocated on the
 * heap
 * @param size The total number of elements of the allocated std::vector data
 * structure which are instantiated
 * @return Success or failure
 */
template <class T>
bool allocateMemoryIfNonRealTime(
    bool is_real_time, std::unique_ptr<std::vector<T> >& smart_pointer,
    uint size = 0) {
  // input checking:
  // (we are only allowed to allocate memory on heap if we operate in
  // NON-real-time mode!!!)
  if (is_real_time == true) {
    return false;
  }
  if (size < 0) {
    return false;
  }

  // execute memory allocation on the heap:
  smart_pointer.reset(new std::vector<T>(size));

  // check if the memory allocation on the heap is successful or not:
  if (isMemoryAllocated(smart_pointer, size) == false) {
    return false;
  }

  return true;
}

}  // namespace dmp
#endif
