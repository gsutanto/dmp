#include "dmp/utility/RealTimeAssertor.h"

namespace dmp {

/****************** Start of RealTimeAssertor Definitions ******************/

RealTimeAssertor::RealTimeAssertor() {
  rt_err_file_path.reset(new char[ERROR_STRING_LENGTH]);
  sprintf(rt_err_file_path.get(), "");
}

RealTimeAssertor::RealTimeAssertor(const char* rt_error_file_path_cstr) {
  rt_err_file_path.reset(new char[ERROR_STRING_LENGTH]);
  std::filesystem::path rt_error_file_path(rt_error_file_path_cstr);

  // Create the parent directory of rt_error_file_path_cstr, if it doesn't exist
  // yet:
  createDirIfNotExistYet(rt_error_file_path.parent_path().string().c_str());

  strcpy(rt_err_file_path.get(), rt_error_file_path_cstr);
}

bool RealTimeAssertor::isValid() {
  if (rt_err_file_path == NULL) {
    return false;
  }

  return true;
}

void RealTimeAssertor::clear_rt_err_file() {
  FILE* fp;

  fp = fopen(rt_err_file_path.get(), "w");
  if (fp != NULL) {
    fclose(fp);
  }
}

void RealTimeAssertor::appendStringTo_rt_err_file(
    struct ErrorStringContainer err_str_container) {
  FILE* fp;

  fp = fopen(rt_err_file_path.get(), "a");
  if (fp != NULL) {
    rt_err_file_mutex.lock();
    fprintf(fp, "%s", err_str_container.error_string);
    rt_err_file_mutex.unlock();
    fclose(fp);
  }
}

RealTimeAssertor::~RealTimeAssertor() {}

/****************** End of RealTimeAssertor Definitions ******************/

void callAppendStringTo_rt_err_file(
    RealTimeAssertor* real_time_assertor_ptr,
    struct ErrorStringContainer err_str_container) {
  real_time_assertor_ptr->appendStringTo_rt_err_file(err_str_container);
}

bool realTimeAssertFailed(RealTimeAssertor* real_time_assertor_ptr,
                          const char* expression_string, const char* file_name,
                          uint line_number) {
  if (real_time_assertor_ptr == NULL) {
    return false;
  }

  struct ErrorStringContainer assertion_err_str_container;

  sprintf(assertion_err_str_container.error_string,
          "Assertion %s is failed at file %s, line %u.\n", expression_string,
          file_name, line_number);

  // Spawn a non-real-time-thread with minimum priority (such that it can be
  // preempted/yanked by any real-time thread or any thread with higher
  // priority) to execute the appending of the error message onto the file
  // specified in the RealTimeAssertor class:
  int non_rt_thread_policy = SCHED_OTHER;  // non-real-time-thread
  struct sched_param non_rt_thread_param;
  non_rt_thread_param.sched_priority = sched_get_priority_min(
      non_rt_thread_policy);  // minimum-priority of the thread policy specified
                              // by non_rt_thread_policy

  std::thread non_rt_thread(callAppendStringTo_rt_err_file,
                            real_time_assertor_ptr,
                            assertion_err_str_container);

  pthread_setschedparam(non_rt_thread.native_handle(), non_rt_thread_policy,
                        &non_rt_thread_param);
  non_rt_thread.detach();

  return false;
}

}  // namespace dmp
