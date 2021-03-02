#include "dmp/paths.h"

#include <cstring>

// DMP_DATA_DIR, DMP_PLOT_DIR and DMP_RT_ERRORS_DIR are set to correct absolute
// path during compile time. See CMakeLists.txt.

std::string get_data_path(const char* relative_path) {
  return std::string(DMP_DATA_DIR + std::string(relative_path));
}

void get_data_path(char* get, const char* relative_path) {
  std::string str_abs_path = get_data_path(relative_path);
  strcpy(get, str_abs_path.c_str());
}

std::string get_matlab_path(const char* relative_path) {
  return std::string(DMP_MATLAB_DIR + std::string(relative_path));
}

void get_matlab_path(char* get, const char* relative_path) {
  std::string str_abs_path = get_matlab_path(relative_path);
  strcpy(get, str_abs_path.c_str());
}

std::string get_python_path(const char* relative_path) {
  return std::string(DMP_PYTHON_DIR + std::string(relative_path));
}

void get_python_path(char* get, const char* relative_path) {
  std::string str_abs_path = get_python_path(relative_path);
  strcpy(get, str_abs_path.c_str());
}

std::string get_plot_path(const char* relative_path) {
  return std::string(DMP_PLOT_DIR + std::string(relative_path));
}

void get_plot_path(char* get, const char* relative_path) {
  std::string str_abs_path = get_plot_path(relative_path);
  strcpy(get, str_abs_path.c_str());
}

std::string get_rt_errors_path(const char* relative_path) {
  return std::string(DMP_RT_ERRORS_DIR + std::string(relative_path));
}

void get_rt_errors_path(char* get, const char* relative_path) {
  std::string str_abs_path = get_rt_errors_path(relative_path);
  strcpy(get, str_abs_path.c_str());
}
