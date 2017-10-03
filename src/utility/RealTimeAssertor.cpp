#include "amd_clmc_dmp/utility/RealTimeAssertor.h"

namespace dmp
{

    /************************ Start of RealTimeAssertor Definitions ************************/

    RealTimeAssertor::RealTimeAssertor()
    {
        rt_err_file_path.reset(new char[ERROR_STRING_LENGTH]);
        sprintf(rt_err_file_path.get(), "");
    }

    /**
     * @param rt_error_file_path The path to the real-time-assertion-error-logging file
     */
    RealTimeAssertor::RealTimeAssertor(const char* rt_error_file_path)
    {
        rt_err_file_path.reset(new char[ERROR_STRING_LENGTH]);
        boost::filesystem::path rt_error_boost_file_path(rt_error_file_path);

        // Create the parent directory of rt_error_file_path, if it doesn't exist yet:
        createDirIfNotExistYet(rt_error_boost_file_path.parent_path().string().c_str());

        strcpy(rt_err_file_path.get(), rt_error_file_path);
    }

    /**
     * @return RealTimeAssertor is valid (true) or invalid (false)
     */
    bool RealTimeAssertor::isValid()
    {
        if (rt_err_file_path == NULL)
        {
            return false;
        }

        return true;
    }

    /**
     * Empty out the real-time-error-log file specified in the RealTimeAssertor class (for refilling).
     */
    void RealTimeAssertor::clear_rt_err_file()
    {
        FILE*   fp;

        fp      = fopen(rt_err_file_path.get(), "w");
        if (fp != NULL)
        {
            fclose(fp);
        }
    }

    /**
     * Append the error_string contained in the ErrorStringContainer struct into
     * the real-time-error-log file specified in the RealTimeAssertor class.
     *
     * @param err_str_container The ErrorStringContainer struct containing the error_string to be appended
     */
    void RealTimeAssertor::appendStringTo_rt_err_file(struct ErrorStringContainer err_str_container)
    {
        FILE*           fp;

        fp      = fopen(rt_err_file_path.get(), "a");
        if (fp != NULL)
        {
            rt_err_file_mutex.lock();
            fprintf(fp, "%s", err_str_container.error_string);
            rt_err_file_mutex.unlock();
            fclose(fp);
        }
    }

    RealTimeAssertor::~RealTimeAssertor()
    {}

    /************************ End of RealTimeAssertor Definitions ************************/

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
                                        struct ErrorStringContainer err_str_container)
    {
        real_time_assertor_ptr->appendStringTo_rt_err_file(err_str_container);
    }

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
                              uint line_number)
    {
        if (real_time_assertor_ptr == NULL)
        {
            return false;
        }

        struct ErrorStringContainer assertion_err_str_container;

        sprintf(assertion_err_str_container.error_string, "Assertion %s is failed at file %s, line %u.\n",
                expression_string, file_name, line_number);

        // Spawn a non-real-time-thread with minimum priority (such that it can be preempted/yanked by
        // any real-time thread or any thread with higher priority) to execute
        // the appending of the error message onto the file specified in the RealTimeAssertor class:
        int                         non_rt_thread_policy        = SCHED_OTHER;  // non-real-time-thread
        struct sched_param 			non_rt_thread_param;
        non_rt_thread_param.sched_priority                      = sched_get_priority_min(non_rt_thread_policy); // minimum-priority of the thread policy specified by non_rt_thread_policy

        boost::thread   non_rt_thread(callAppendStringTo_rt_err_file,
                                      real_time_assertor_ptr, assertion_err_str_container);

        pthread_setschedparam(non_rt_thread.native_handle(), non_rt_thread_policy, &non_rt_thread_param);
        non_rt_thread.detach();

        return false;
    }

}
