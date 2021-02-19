#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>

#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.h"
#include "dmp/paths.h"

using namespace dmp;

void print_usage()
{
    printf("Usage: dmp_cart_coord_dmp_single_traj_training_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-h time_goal_change(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]\n");
}

int main(int argc, char** argv)
{
    bool    is_real_time            = false;

    double  task_servo_rate         = 420.0;    // MasterArm robot's task servo rate
    double  dt                      = 1.0/task_servo_rate;
    uint    model_size              = 50;
    double  tau                     = MIN_TAU;

    double                          tau_learn;
    Trajectory                      critical_states_learn(2);

    uint    formula_type            = _SCHAAL_DMP_;
    uint    canonical_order         = 2;        // default is using 2nd order canonical system
    uint    learning_method         = _SCHAAL_LWR_METHOD_;
    double  time_reproduce_max      = 0.0;
    double  time_goal_change        = 0.0;
    double  tau_reproduce           = 0.0;
    char    dmp_plot_dir_path[1000];
    get_plot_path(dmp_plot_dir_path,"/cart_dmp/cart_coord_dmp/single_traj_training/");
    char    rt_err_file_path[1000];
    get_rt_errors_path(rt_err_file_path,"/rt_err.txt");

    static  struct  option  long_options[]          = {
        {"formula_type",        required_argument, 0,  'f' },
        {"canonical_order",     required_argument, 0,  'c' },
        {"learning_method",     required_argument, 0,  'm' },
        {"time_reproduce_max",  required_argument, 0,  'r' },
        {"time_goal_change",    required_argument, 0,  'h' },
        {"tau_reproduce",       required_argument, 0,  't' },
        {"rt_err_file_path",    required_argument, 0,  'e' },
        {0,                     0,                 0,  0   }
    };

    int     opt                     = 0;
    int     long_index              = 0;
    while ((opt = getopt_long(argc, argv,"f:c:m:r:h:t:e:", long_options, &long_index )) != -1)
    {
        switch (opt)
        {
            case 'f'    : formula_type              = atoi(optarg);
                break;
            case 'c'    : canonical_order           = atoi(optarg);
                break;
            case 'm'    : learning_method           = atoi(optarg);
                break;
            case 'r'    : time_reproduce_max        = atof(optarg);
                break;
            case 'h'    : time_goal_change          = atof(optarg);
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
        (time_reproduce_max < 0.0) || (time_goal_change < 0.0) || (tau_reproduce < 0.0) ||
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

    // Initialize Cartesian Coordinate DMP
    CartesianCoordDMP       cart_dmp(&canonical_sys_discr, model_size,
                                     formula_type, learning_method,
                                     &rt_assertor, _GSUTANTO_LOCAL_COORD_FRAME_, dmp_plot_dir_path);

    // Learn the Cartesian trajectory
    if (rt_assert_main(cart_dmp.learn(get_data_path("/cart_dmp/cart_coord_dmp/single_traj_training/sample_traj_3D_1.txt").c_str(),
                                      task_servo_rate, &tau_learn, &critical_states_learn)) == false)
    {
        return (-1);
    }

    /**** Reproduce ****/

    if (time_reproduce_max <= 0.0)
    {
        time_reproduce_max		= tau_learn;
    }

    if (time_goal_change <= 0.0)
    {
        time_goal_change        = time_reproduce_max;
    }

    if (tau_reproduce <= 0.0)
    {
        tau_reproduce			= tau_learn;
    }

    // change tau here during movement reproduction:
    tau                         = tau_reproduce;

    // define unrolling parameters:
    DMPUnrollInitParams         dmp_unroll_init_parameters(tau, critical_states_learn,
                                                           &rt_assertor);

    VectorN                     steady_state_goal_position(3);
    steady_state_goal_position.block(0,0,3,1)   << 0.5, 1.0, 0.0;

    // Start DMP
    if (rt_assert_main(cart_dmp.start(dmp_unroll_init_parameters)) == false)
    {
        return (-1);
    }

    // Run DMP and print state values:
    if (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet()) == false)
    {
        return (-1);
    }
    for (uint i = 0; i < (round(time_reproduce_max*task_servo_rate) + 1); ++i)
    {
        double  time = 1.0 * (i*dt);

        // Get the next state of the Cartesian DMP
        DMPState current_state(3, &rt_assertor);

        if (rt_assert_main(cart_dmp.getNextState(dt, true, current_state)) == false)
        {
            return (-1);
        }

        double epsilon  = std::numeric_limits<double>::epsilon();
        if (fabs(time - time_goal_change) < (5 * epsilon))    // time to execute goal change
        {
            Vector3     steady_state_goal_position;
            steady_state_goal_position  << 0.5, 0.5, 0.0;
            if (rt_assert_main(cart_dmp.setNewSteadyStateGoalPosition(steady_state_goal_position)) == false)
            {
                return (-1);
            }
        }

        // Log state trajectory:
        printf("%.05f %.05f %.05f %.05f\n", time, current_state.getX()[0], current_state.getX()[1], current_state.getX()[2]);
    }
    cart_dmp.stopCollectingTrajectoryDataSet();
    if (rt_assert_main(cart_dmp.saveTrajectoryDataSet()) == false)
    {
        return (-1);
    }

    return 0;
}
