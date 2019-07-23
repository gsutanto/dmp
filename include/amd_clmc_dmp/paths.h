#ifndef PATHS_H
#define PATHS_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string>
#include <cstring>

using namespace std;

std::string get_data_path(const char* relative_path);
void get_data_path(char * get, const char* relative_path);
std::string get_matlab_path(const char* relative_path);
void get_matlab_path(char * get, const char* relative_path);
std::string get_python_path(const char* relative_path);
void get_python_path(char * get, const char* relative_path);
std::string get_plot_path(const char* relative_path);
void get_plot_path(char * get, const char* relative_path);
std::string get_rt_errors_path(const char* relative_path);
void get_rt_errors_path(char * get, const char* relative_path);
#endif
