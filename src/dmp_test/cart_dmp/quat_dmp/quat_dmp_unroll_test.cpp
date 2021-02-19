#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>

#include "dmp/dmp_state/QuaternionDMPState.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/cart_dmp/quat_dmp/QuaternionDMP.h"
#include "dmp/utility/DataIO.h"
#include "dmp/paths.h"

using namespace dmp;

void print_usage()
{
    printf("Usage: dmp_quat_dmp_unroll_demo [-r time_reproduce_max(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]\n");
}

int main(int argc, char** argv)
{
    bool    is_real_time            = false;

    double  task_servo_rate         = 300.0;    // ARM robot's task servo rate
    double  dt                      = 1.0/task_servo_rate;
    uint    model_size              = 25;
    double  tau                     = MIN_TAU;

    double                          tau_learn;
    QuaternionTrajectory            critical_states_learn(2);

    uint    formula_type            = _SCHAAL_DMP_;
    uint    canonical_order         = 2;        // default is using 2nd order canonical system
    uint    learning_method         = _SCHAAL_LWR_METHOD_;
    double  time_reproduce_max      = 0.0;
    double  tau_reproduce           = 0.0;
    char    dmp_plot_dir_path[1000];
    get_plot_path(dmp_plot_dir_path,"/cart_dmp/quat_dmp/");
    char    rt_err_file_path[1000];
    get_rt_errors_path(rt_err_file_path,"/rt_err.txt");

    static  struct  option  long_options[]          = {
        {"time_reproduce_max",  required_argument, 0,  'r' },
        {"tau_reproduce",       required_argument, 0,  't' },
        {"rt_err_file_path",    required_argument, 0,  'e' },
        {0,                     0,                 0,  0   }
    };

    int     opt                     = 0;
    int     long_index              = 0;
    while ((opt = getopt_long(argc, argv,"r:t:e:", long_options, &long_index )) != -1)
    {
        switch (opt)
        {
            case 'r'    : time_reproduce_max        = atof(optarg);
                break;
            case 't'    : tau_reproduce             = atof(optarg);
                break;
            case 'e'    : strcpy(rt_err_file_path, optarg);
                break;
            default: print_usage();
                exit(EXIT_FAILURE);
        }
    }

    if ((formula_type < _SCHAAL_DMP_) || (formula_type > _HOFFMANN_DMP_) ||
        (canonical_order < 1) || (canonical_order > 2) ||
        (learning_method < _SCHAAL_LWR_METHOD_) || (learning_method > _JPETERS_GLOBAL_LEAST_SQUARES_METHOD_) ||
        (time_reproduce_max < 0.0) || (tau_reproduce < 0.0) ||
        (strcmp(rt_err_file_path, "") == 0))
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    // Initialize real-time assertor
    RealTimeAssertor        rt_assertor(rt_err_file_path);
    rt_assertor.clear_rt_err_file();

    // Initialize tau system
    TauSystem               tau_sys(MIN_TAU, &rt_assertor);

    // Initialize canonical system
    CanonicalSystemDiscrete canonical_sys_discr(&tau_sys, &rt_assertor, canonical_order);

    // Initialize Quaternion DMP
    QuaternionDMP           quat_dmp(model_size, &canonical_sys_discr,
                                     learning_method, &rt_assertor, dmp_plot_dir_path);

    // Load the Quaternion DMP parameters
    if (rt_assert_main(quat_dmp.loadParams(get_data_path("/cart_dmp/quat_dmp/").c_str(),
                                           "w", "A_learn", "Q0", "QG", "tau")) == false)
    {
        return (-1);
    }
    tau_learn               = quat_dmp.getMeanTau();
    critical_states_learn[0]= QuaternionDMPState(quat_dmp.getMeanStartPosition(), &rt_assertor);
    if (rt_assert_main(critical_states_learn[0].computeQdAndQdd()) == false)
    {
        return (-1);
    }
    critical_states_learn[1]= QuaternionDMPState(quat_dmp.getMeanGoalPosition(), &rt_assertor);
    if (rt_assert_main(critical_states_learn[1].computeQdAndQdd()) == false)
    {
        return (-1);
    }

    /**** Unrolling Time Parameters ****/

    if (time_reproduce_max <= 0.0)
    {
        time_reproduce_max		= tau_learn;
    }

    if (tau_reproduce <= 0.0)
    {
        tau_reproduce			= tau_learn;
    }

    // change tau here during movement reproduction:
    tau                         = tau_reproduce;

    /***** Start Unrolling Loaded Quaternion DMP with Zero Initial Angular Velocity and Angular Acceleration *****/

    // Start Quaternion DMP
    if (rt_assert_main(quat_dmp.startQuaternionDMP(critical_states_learn, tau)) == false)
    {
        return (-1);
    }

    // Run DMP and print state values:
    for (uint i = 0; i < (round(time_reproduce_max*task_servo_rate) + 1); ++i)
    {
        double  time = 1.0 * (i*dt);

        // Get the next state of the Quaternion DMP
        QuaternionDMPState current_state(&rt_assertor);

        if (rt_assert_main(quat_dmp.getNextQuaternionState(dt, true, current_state)) == false)
        {
            return (-1);
        }

        // Log state trajectory:
        printf("%.05f %.05f %.05f %.05f %.05f\n",
               time,
               current_state.getQ()[0], current_state.getQ()[1], current_state.getQ()[2], current_state.getQ()[3]);
    }

    /***** End Unrolling Loaded Quaternion DMP with Zero Initial Angular Velocity and Angular Acceleration *****/

    DataIO                  data_io(&rt_assertor);
    // Load additional unrolling test parameters
    Vector3                 nonzero_omega           = ZeroVector3;
    Vector3                 nonzero_omegad          = ZeroVector3;
    if (rt_assert_main(data_io.readMatrixFromFile(get_data_path("/cart_dmp/quat_dmp/").c_str(),
                                                  "nonzero_omega", nonzero_omega)) == false)
    {
        return (-1);
    }
    if (rt_assert_main(data_io.readMatrixFromFile(get_data_path("/cart_dmp/quat_dmp/").c_str(),
                                                  "nonzero_omegad", nonzero_omegad)) == false)
    {
        return (-1);
    }

    if (rt_assert_main(critical_states_learn[0].setOmega(nonzero_omega)) == false)
    {
        return (-1);
    }
    if (rt_assert_main(critical_states_learn[0].setOmegad(nonzero_omegad)) == false)
    {
        return (-1);
    }
    if (rt_assert_main(critical_states_learn[0].computeQdAndQdd()) == false)
    {
        return (-1);
    }
    // unroll at a longer duration:
    tau                 = 1.5 * tau;
    time_reproduce_max  = 1.5 * time_reproduce_max;

    /***** Start Unrolling Loaded Quaternion DMP with NON-Zero Initial Angular Velocity and Angular Acceleration *****/

    // Start Quaternion DMP
    if (rt_assert_main(quat_dmp.startQuaternionDMP(critical_states_learn, tau)) == false)
    {
        return (-1);
    }

    // Run DMP and print state values:
    for (uint i = 0; i < (round(time_reproduce_max*task_servo_rate) + 1); ++i)
    {
        double  time = 1.0 * (i*dt);

        // Get the next state of the Quaternion DMP
        QuaternionDMPState current_state(&rt_assertor);

        if (rt_assert_main(quat_dmp.getNextQuaternionState(dt, true, current_state)) == false)
        {
            return (-1);
        }

        // Log state trajectory:
        printf("%.05f %.05f %.05f %.05f %.05f\n",
               time,
               current_state.getQ()[0], current_state.getQ()[1], current_state.getQ()[2], current_state.getQ()[3]);
    }

    /***** End Unrolling Loaded Quaternion DMP with NON-Zero Initial Angular Velocity and Angular Acceleration *****/

    return 0;
}
