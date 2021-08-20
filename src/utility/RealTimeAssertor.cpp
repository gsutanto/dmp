#include "dmp/utility/RealTimeAssertor.h"

namespace dmp {

/****************** Start of RealTimeAssertor Definitions ******************/

RealTimeAssertor::RealTimeAssertor() {
  rt_err_file_path.reset(new char[ERROR_STRING_LENGTH]);
  snprintf(rt_err_file_path.get(), ERROR_STRING_LENGTH, "");
  error_log.clear();
  error_log.reserve(NUM_MAX_ERROR_STRINGS);
}

RealTimeAssertor::RealTimeAssertor(const char* rt_error_file_path_cstr) {
  rt_err_file_path.reset(new char[ERROR_STRING_LENGTH]);
  snprintf(rt_err_file_path.get(), ERROR_STRING_LENGTH, "");
  std::filesystem::path rt_error_file_path(rt_error_file_path_cstr);
  error_log.clear();
  error_log.reserve(NUM_MAX_ERROR_STRINGS);

  // Create the parent directory of rt_error_file_path_cstr, if it doesn't exist
  // yet:
  createDirIfNotExistYet(rt_error_file_path.parent_path().string().c_str());

  snprintf(rt_err_file_path.get(), ERROR_STRING_LENGTH, "%s",
           rt_error_file_path_cstr);
}

bool RealTimeAssertor::isValid() {
  if (rt_err_file_path == nullptr) {
    return false;
  }

  return true;
}

void RealTimeAssertor::appendErrorStringToInternalErrorLog(
    struct ErrorStringContainer err_str_container) {
  rt_err_file_mutex.lock();
  error_log.push_back(err_str_container);
  rt_err_file_mutex.unlock();
}

void RealTimeAssertor::clear_rt_err_file() {
  FILE* fp;

  fp = fopen(rt_err_file_path.get(), "w");
  if (fp != nullptr) {
    fclose(fp);
  }
}

void RealTimeAssertor::writeErrorsTo_rt_err_file() {
  FILE* fp;

  fp = fopen(rt_err_file_path.get(), "w");
  if (fp != nullptr) {
    for (const auto& err_str_container : error_log) {
      fprintf(fp, "%s", err_str_container.error_string);
    }
    fclose(fp);
  }
}

int RealTimeAssertor::writeErrorsAndReturnIntErrorCode(int error_code) {
  writeErrorsTo_rt_err_file();
  return error_code;
}

void RealTimeAssertor::getErrorLog(
    std::vector<ErrorStringContainer>& returned_error_log) {
  returned_error_log = error_log;
}

RealTimeAssertor::~RealTimeAssertor() {}

/****************** End of RealTimeAssertor Definitions ******************/

bool realTimeAssertFailed(RealTimeAssertor* real_time_assertor_ptr,
                          const char* expression_string, const char* file_name,
                          uint line_number) {
  if (real_time_assertor_ptr == nullptr) {
    return false;
  }

  struct ErrorStringContainer assertion_err_str_container;

  snprintf(assertion_err_str_container.error_string,
           sizeof(assertion_err_str_container.error_string),
           "Assertion %s is failed at file %s, line %u.\n", expression_string,
           file_name, line_number);

  real_time_assertor_ptr->appendErrorStringToInternalErrorLog(
      assertion_err_str_container);

  return false;
}

}  // namespace dmp
