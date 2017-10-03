/*
 * RealTimeAssertor.h
 *
 *  Provides a class for real-time assertion, by spawning a non-real-time thread when the assertion is failed,
 *  to deal with assertion error logging into a file.\n\n
 *  This will be VERY useful for troubleshooting and debugging.
 *
 *  Created on: December 17, 2015
 *  Author: Giovanni Sutanto
 */

#ifndef REAL_TIME_ASSERTOR_H
#define REAL_TIME_ASSERTOR_H

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "amd_clmc_dmp/utility/utility.h"


namespace dmp
{

/*! defines */
/**
 * Real-Time Assertion\n
 * NOTE: This must ONLY be called from methods of a class \n
 *       that has a pointer to a RealTimeAssertor instance as a field of the class.\n
 * \n
 * IF expr is evaluated as 'true', then do nothing.\n
 * (ELSE) IF expr is evaluated as 'false', then spawn-off a non-real-time thread that
 * will print off the error message onto a file.
 */
#define rt_assert(expr) \
  ((expr)               \
   ? (true)             \
   : (realTimeAssertFailed(this->rt_assertor, __STRING(expr), __FILE__, __LINE__)))

/**
 * In main-function Real-Time Assertion\n
 * Same as rt_assert, but to be called from the main function/program.\n
 * NOTE: To use this in any main function/program, the following definition shall be made BEFOREHAND:\n
 *       RealTimeAssertor                rt_assertor(rt_err_file_path);
 */
#define rt_assert_main(expr)            \
  ((expr)                               \
   ? (true)                             \
   : (realTimeAssertFailed(&rt_assertor, __STRING(expr), __FILE__, __LINE__)))

#define ERROR_STRING_LENGTH 10000

/**
 * Struct ErrorStringContainer is a wrapper of the error_string, to ensure that
 * each non-real-time-thread have a local/private (NOT shared) copy of it.
 */
struct ErrorStringContainer
{
    char            error_string[ERROR_STRING_LENGTH];
};

class RealTimeAssertor
{

private:

    // The following field(s) are written as SMART arrays to reduce size overhead
    // in the stack, by allocating them in the heap:
    CharArr         rt_err_file_path;

    boost::mutex    rt_err_file_mutex;

public:

    RealTimeAssertor();

    /**
     * @param rt_error_file_path The path to the real-time-assertion-error-logging file
     */
    RealTimeAssertor(const char* rt_error_file_path);

    /**
     * @return RealTimeAssertor is valid (true) or invalid (false)
     */
    bool isValid();

    /**
     * Empty out the real-time-error-log file specified in the RealTimeAssertor class (for refilling).
     */
    void clear_rt_err_file();

    /**
     * Append the error_string contained in the ErrorStringContainer struct into
     * the real-time-error-log file specified in the RealTimeAssertor class.
     *
     * @param err_str_container The ErrorStringContainer struct containing the error_string to be appended
     */
    void appendStringTo_rt_err_file(struct ErrorStringContainer err_str_container);

    ~RealTimeAssertor();

};

/**
 * Wrapper function to call the appendStringTo_rt_err_file() function inside
 * the RealTimeAssertor class on the specified ErrorStringContainer struct.
 *
 * @param real_time_assertor_ptr Pointer to the RealTimeAssertor class, \n
 *                               whose appendStringTo_rt_err_file() function is to be called
 * @param err_str_container The ErrorStringContainer struct, which is \n
 *                          the input to the appendStringTo_rt_err_file() function
 */
void callAppendStringTo_rt_err_file(RealTimeAssertor* real_time_assertor_ptr,
                                    struct ErrorStringContainer err_str_container);

/**
 * Real-Time Assertion is Failed\n
 * Thus this will spawn-off a non-real-time thread that will append the error message onto
 * the file specified in the RealTimeAssertor class.
 *
 * @param expression_string Assertion expression that was evaluated as 'false', expressed as a C-string
 * @param file_name File name where the assertion was failed
 * @param line_number Line number on the file where the assertion was failed
 * @return Success or Failure
 */
bool realTimeAssertFailed(RealTimeAssertor* real_time_assertor_ptr,
                          const char* expression_string,
                          const char* file_name,
                          uint line_number);

}

#endif
