#include "dmp/neural_nets/PMNN.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

#include "dmp/paths.h"

using namespace dmp;

void print_usage() { printf("Usage: dmp_pmnn_demo [-e rt_err_file_path]\n"); }

int main(int argc, char** argv) {
  bool is_real_time = false;

  std::vector<uint> topology(4);
  topology[0] = 45;
  topology[1] = 100;  // just an initialization, may be changed later by loading
                      // from a file
  topology[2] = 25;
  topology[3] = 1;

  std::vector<uint> activation_functions(4);
  activation_functions[0] = _IDENTITY_;
  activation_functions[1] = _TANH_;
  activation_functions[2] = _IDENTITY_;
  activation_functions[3] = _IDENTITY_;

  char rt_err_file_path[1000];
  std::string rt_err_file_path_str = 
    get_rt_errors_path("/rt_err.txt");
  snprintf(rt_err_file_path, sizeof(rt_err_file_path),
           "%s", rt_err_file_path_str.c_str());

  static struct option long_options[] = {
      {"rt_err_file_path", required_argument, 0, 'e'}, {0, 0, 0, 0}};

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "e:", long_options, &long_index)) !=
         -1) {
    switch (opt) {
      case 'e':
        snprintf(rt_err_file_path, sizeof(rt_err_file_path), "%s", optarg);
        break;
      default:
        print_usage();
        exit(EXIT_FAILURE);
    }
  }

  if (strcmp(rt_err_file_path, "") == 0) {
    print_usage();
    exit(EXIT_FAILURE);
  }

  // Initialize real-time assertor
  RealTimeAssertor rt_assertor(rt_err_file_path);
  rt_assertor.clear_rt_err_file();

  DataIO data_io(&rt_assertor);

  int regular_NN_hidden_layer_topology = 0;
  if (rt_assert_main(data_io.readMatrixFromFile(
          get_python_path("/dmp_coupling/learn_tactile_feedback/models/")
              .c_str(),
          "regular_NN_hidden_layer_topology.txt",
          regular_NN_hidden_layer_topology)) == false) {
    printf("Failed reading PMNN topology specification\n");
    return rt_assertor.writeErrorsAndReturnIntErrorCode(-1);
  }
  topology[1] = (uint)regular_NN_hidden_layer_topology;

  PMNN pmnn(6, topology, activation_functions, &rt_assertor);

  if (rt_assert_main(pmnn.loadParams(
          get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/"
                        "neural_nets/pmnn/cpp_models/prim1/")
              .c_str(),
          0)) == false) {
    return rt_assertor.writeErrorsAndReturnIntErrorCode(-1);
  }

  MatrixXxXPtr X;
  if (rt_assert_main(data_io.readMatrixFromFile(
          get_data_path(
              "/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/pmnn/"
              "unroll_test_dataset/test_unroll_prim_1_X_raw_scraping.txt")
              .c_str(),
          X)) == false) {
    return rt_assertor.writeErrorsAndReturnIntErrorCode(-1);
  }

  MatrixXxXPtr normalized_phase_PSI_mult_phase_V;
  if (rt_assert_main(data_io.readMatrixFromFile(
          get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/"
                        "neural_nets/pmnn/unroll_test_dataset/"
                        "test_unroll_prim_1_normalized_phase_PSI_mult_phase_V_"
                        "scraping.txt")
              .c_str(),
          normalized_phase_PSI_mult_phase_V)) == false) {
    return rt_assertor.writeErrorsAndReturnIntErrorCode(-1);
  }

  uint Nri = X->rows();
  uint Nci = X->cols();
  uint Nrp = normalized_phase_PSI_mult_phase_V->rows();
  uint Ncp = normalized_phase_PSI_mult_phase_V->cols();
  if (rt_assert_main(Nri == Nrp) == false) {
    return rt_assertor.writeErrorsAndReturnIntErrorCode(-1);
  }

  VectorNN_N input(Nci), phase_kernel_modulation(Ncp), output(6);
  input = ZeroVectorNN_N(Nci);
  phase_kernel_modulation = ZeroVectorNN_N(Ncp);
  output = ZeroVectorNN_N(6);
  for (uint t = 0; t < Nri; ++t) {
    input = X->block(t, 0, 1, Nci).transpose();
    phase_kernel_modulation =
        normalized_phase_PSI_mult_phase_V->block(t, 0, 1, Ncp).transpose();
    if (rt_assert_main(pmnn.computePrediction(input, phase_kernel_modulation,
                                              output)) == false) {
      printf("Failed making prediction!");
      return rt_assertor.writeErrorsAndReturnIntErrorCode(-1);
    }

    printf("%.05f %.05f %.05f %.05f %.05f %.05f\n", output(0, 0), output(1, 0),
           output(2, 0), output(3, 0), output(4, 0), output(5, 0));
  }

  return 0;
}
