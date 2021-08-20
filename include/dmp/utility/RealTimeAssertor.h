/*
 * RealTimeAssertor.h
 *
 *  Provides a class for real-time assertion, by spawning a non-real-time thread
 * when the assertion is failed, to deal with assertion error logging into a
 * file.\n\n This will be VERY useful for troubleshooting and debugging.
 *
 *  Created on: December 17, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef REAL_TIME_ASSERTOR_H
#define REAL_TIME_ASSERTOR_H

#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <mutex>

#include "dmp/utility/utility.h"

namespace dmp {

/*! defines */
/**
 * Real-Time Assertion\n
 * NOTE: This must ONLY be called from methods of a class \n
 *       that has a pointer to a RealTimeAssertor instance as a field of the \n
 *       class.\n
 *       \n
 *       IF expr is evaluated as 'true', then do nothing.\n
 *       (ELSE) IF expr is evaluated as 'false', then log the error string \n
 *       into the buffer.
 */
#define rt_assert(expr)                                                        \
  ((expr) ? (true)                                                             \
          : (realTimeAssertFailed(this->rt_assertor, __STRING(expr), __FILE__, \
                                  __LINE__)))

/**
 * In main-function Real-Time Assertion\n
 * Same as rt_assert, but to be called from the main function/program.\n
 * NOTE: To use this in any main function/program, the following definition
 * shall be made BEFOREHAND:\n RealTimeAssertor rt_assertor(rt_err_file_path);
 */
#define rt_assert_main(expr)                                              \
  ((expr) ? (true)                                                        \
          : (realTimeAssertFailed(&rt_assertor, __STRING(expr), __FILE__, \
                                  __LINE__)))

#define ERROR_STRING_LENGTH 10000
#define NUM_MAX_ERROR_STRINGS 20

/**
 * Struct ErrorStringContainer is a wrapper of the error_string.
 */
struct ErrorStringContainer {
  char error_string[ERROR_STRING_LENGTH];
};

/**
 * Real-time-safe assertion that pre-allocates a buffer for error messages
 * during the initial non-real-time context (happening at its creation),
 * log error messages as detected in the buffer in real-time,
 * and finally during its destruction (i.e. a non-real-time context) can be
 * asked to either log the error messages into a file or to transfer/return the
 * error messages to another logging utility.
 */
class RealTimeAssertor {
 private:
  // only used when (rt_assert_mode == _WRITE_ERRORS_TO_FILE_)
  CharArr rt_err_file_path;

  std::mutex rt_err_file_mutex;

  std::vector<ErrorStringContainer> error_log;

 public:
  RealTimeAssertor();

  /**
   * @param rt_error_file_path The path to the real-time-assertion-error-logging
   * file
   */
  RealTimeAssertor(const char* rt_error_file_path);

  /**
   * @return RealTimeAssertor is valid (true) or invalid (false)
   */
  bool isValid();

  /**
   * Append the error string into the internal buffer/list/vector of errors so
   * far, to be later written into a file or passed to the another error logging
   * utility tool.
   *
   * @param err_str_container The ErrorStringContainer struct containing the
   * error_string
   */
  void appendErrorStringToInternalErrorLog(
      struct ErrorStringContainer err_str_container);

  /**
   * (Non Real-Time) Empty out the real-time-error-log file specified in the
   * RealTimeAssertor class (for refilling).
   */
  void clear_rt_err_file();

  /**
   * Write errors from the internal buffer/list/vector of errors into the
   * real-time-error-log file.
   */
  void writeErrorsTo_rt_err_file();

  /**
   * Write errors from the internal buffer/list/vector of errors into the
   * real-time-error-log file, and then return with the specified error code.
   *
   * @param error_code The (integer) error code to be returned.
   */
  int writeErrorsAndReturnIntErrorCode(int error_code);

  void getErrorLog(std::vector<ErrorStringContainer>& returned_error_log);

  ~RealTimeAssertor();
};

/**
 * Real-Time Assertion is Failed\n
 * Thus this will append the error message --containing the expression string
 * (whose assertion is failed), the file name and the corresponding line number
 * (where the assertion is failed)-- onto the error buffer of the
 * RealTimeAssertor class.
 *
 * @param expression_string Assertion expression that was evaluated as 'false',
 * expressed as a C-string
 * @param file_name File name where the assertion was failed
 * @param line_number Line number on the file where the assertion was failed
 * @return Success or Failure
 */
bool realTimeAssertFailed(RealTimeAssertor* real_time_assertor_ptr,
                          const char* expression_string, const char* file_name,
                          uint line_number);

}  // namespace dmp

#endif
