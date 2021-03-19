#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <ctime>
#include <iostream>

#include "dmp/dmp_1D/DMPDiscrete1D.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_param/TauSystem.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/paths.h"
#include "dmp/utility/DefinitionsDerived.h"

using namespace dmp;

void print_usage() {
  printf(
      "Usage: dmp_1D_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c "
      "canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR "
      "1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] "
      "[-h time_goal_change(>=0)] [-g new_goal] [-t tau_reproduce(>=0)] [-e "
      "rt_err_file_path]\n");
}

int main(int argc, char** argv) {
  bool is_real_time = false;

  double task_servo_rate = 1000.0;  // demonstration/training data frequency
  uint model_size = 25;
  double tau = MIN_TAU;

  double tau_learn;
  Trajectory critical_states_learn(2);

  uint formula_type = _SCHAAL_DMP_;
  uint canonical_order = 2;  // default is using 2nd order canonical system
  uint learning_method = _SCHAAL_LWR_METHOD_;
  double time_reproduce_max = 0.0;
  double time_goal_change = 0.0;
  VectorN new_goal(1);
  new_goal = ZeroVectorN(1);
  double tau_reproduce = 0.0;

  char dmp_plot_dir_path[1000];
  get_plot_path(dmp_plot_dir_path, "/dmp_1D/");

  char rt_err_file_path[1000];
  get_rt_errors_path(rt_err_file_path, "/rt_err.txt");

  static struct option long_options[] = {
      {"formula_type", required_argument, 0, 'f'},
      {"canonical_order", required_argument, 0, 'c'},
      {"learning_method", required_argument, 0, 'm'},
      {"time_reproduce_max", required_argument, 0, 'r'},
      {"time_goal_change", required_argument, 0, 'h'},
      {"new_goal", required_argument, 0, 'g'},
      {"tau_reproduce", required_argument, 0, 't'},
      {"rt_err_file_path", required_argument, 0, 'e'},
      {0, 0, 0, 0}};

  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "f:c:m:r:h:g:t:e:", long_options,
                            &long_index)) != -1) {
    switch (opt) {
      case 'f':
        formula_type = atoi(optarg);
        break;
      case 'c':
        canonical_order = atoi(optarg);
        break;
      case 'm':
        learning_method = atoi(optarg);
        break;
      case 'r':
        time_reproduce_max = atof(optarg);
        break;
      case 'h':
        time_goal_change = atof(optarg);
        break;
      case 'g':
        new_goal[0] = atof(optarg);
        break;
      case 't':
        tau_reproduce = atof(optarg);
        break;
      case 'e':
        snprintf(rt_err_file_path, sizeof(rt_err_file_path), "%s", optarg);
        break;
      default:
        print_usage();
        exit(EXIT_FAILURE);
    }
  }

  if ((formula_type < _SCHAAL_DMP_) || (formula_type > _HOFFMANN_DMP_) ||
      (canonical_order < 1) || (canonical_order > 2) ||
      (learning_method < _SCHAAL_LWR_METHOD_) ||
      (learning_method > _JPETERS_GLOBAL_LEAST_SQUARES_METHOD_) ||
      (time_reproduce_max < 0.0) || (time_goal_change < 0.0) ||
      (tau_reproduce < 0.0) || (strcmp(rt_err_file_path, "") == 0)) {
    print_usage();
    exit(EXIT_FAILURE);
  }

  // Initialize real-time assertor
  RealTimeAssertor rt_assertor(rt_err_file_path);
  rt_assertor.clear_rt_err_file();

  // Initialize tau system
  TauSystem tau_sys(MIN_TAU, &rt_assertor);

  // Initialize canonical system
  CanonicalSystemDiscrete can_sys_discr(&tau_sys, &rt_assertor,
                                        canonical_order);

  // Initialize DMP
  DMPDiscrete1D dmp_discrete_1D(model_size, &can_sys_discr, formula_type,
                                learning_method, &rt_assertor,
                                dmp_plot_dir_path);

  // Learn the trajectory
  if (rt_assert_main(dmp_discrete_1D.learn(
          get_data_path("/dmp_1D/sample_traj_1.txt").c_str(), task_servo_rate,
          &tau_learn, &critical_states_learn)) == false) {
    return (-1);
  }

  /** Store Basis Functions Magnitude over Varying Canonical State Position **/
  /*if
  (rt_assert_main(dmp_discrete_1D.saveBasisFunctions("../plot/dmp_1D/sample_dmp_basis_functions.txt"))
  == false)
  {
      return (-1);
  }*/
  // test weight-saving:
  if (rt_assert_main(dmp_discrete_1D.saveParams(
          dmp_plot_dir_path, "sample_dmp_1D_learned_weights.txt",
          "sample_dmp_1D_A_learn.txt", "sample_dmp_1D_mean_start_position.txt",
          "sample_dmp_1D_mean_goal_position.txt",
          "sample_dmp_1D_mean_tau.txt")) == false) {
    return (-1);
  }
  // test weight-loading:
  if (rt_assert_main(dmp_discrete_1D.loadParams(
          dmp_plot_dir_path, "sample_dmp_1D_learned_weights.txt",
          "sample_dmp_1D_A_learn.txt", "sample_dmp_1D_mean_start_position.txt",
          "sample_dmp_1D_mean_goal_position.txt",
          "sample_dmp_1D_mean_tau.txt")) == false) {
    return (-1);
  }

  /**** Reproduce ****/

  if (time_reproduce_max <= 0.0) {
    time_reproduce_max = tau_learn;
  }

  if (new_goal.isZero() == true) {
    new_goal = dmp_discrete_1D.getMeanGoalPosition();
  }

  if (tau_reproduce <= 0.0) {
    tau_reproduce = tau_learn;
  }

  // change tau here during movement reproduction:
  tau = tau_reproduce;

  // define unrolling parameters:
  DMPUnrollInitParams dmp_unroll_init_params(tau, critical_states_learn,
                                             &rt_assertor);

  // Start DMP
  if (rt_assert_main(dmp_discrete_1D.start(dmp_unroll_init_params)) ==
      false) {
    return (-1);
  }

  double dt = 1.0 / task_servo_rate;

  // Run DMP and print state values:
  if (rt_assert_main(dmp_discrete_1D.startCollectingTrajectoryDataSet()) ==
      false) {
    return (-1);
  }
  // std::clock_t begin = std::clock();
  for (uint i = 0; i < (round(time_reproduce_max * task_servo_rate) + 1); i++) {
    // try changing goal here during movement reproduction:
    if (i == (round(time_goal_change * task_servo_rate) +
              1))  // start changing goal at (i/300) seconds
    {
      if (rt_assert_main(dmp_discrete_1D.setNewSteadyStateGoalPosition(
              new_goal)) == false) {
        return (-1);
      }
    }

    // Get the next state of the DMP
    DMPState state(&rt_assertor);
    if (rt_assert_main(dmp_discrete_1D.getNextState(dt, true, state)) ==
        false) {
      return (-1);
    }

    printf("%.05f %.05f %.05f %.05f\n", state.getTime(), state.getX()[0],
           state.getXd()[0], state.getXdd()[0]);
  }
  // std::clock_t end = std::clock();
  // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  // std::cout << "elapsed_secs = " << elapsed_secs << std::endl;
  dmp_discrete_1D.stopCollectingTrajectoryDataSet();
  if (rt_assert_main(dmp_discrete_1D.saveTrajectoryDataSet()) == false) {
    return (-1);
  }

  return 0;
}
