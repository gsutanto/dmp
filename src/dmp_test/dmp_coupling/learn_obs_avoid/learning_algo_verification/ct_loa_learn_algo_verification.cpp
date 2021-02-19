#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>

#include "dmp/utility/utility.h"
#include "dmp/utility/RealTimeAssertor.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"
#include "dmp/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.h"
#include "dmp/dmp_coupling/learn_obs_avoid/TransformCouplingLearnObsAvoid.h"
#include "dmp/paths.h"

using namespace dmp;

void print_usage()
{
    printf("Usage: dmp_ct_loa_learn_algo_verification_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-o loa_plot_dir_path] [-e rt_err_file_path]\n");
}

int main(int argc, char** argv)
{
    bool        is_real_time                = false;

    double      task_servo_rate             = 420.0;    // MasterArm robot's task servo rate
    double      dt                          = 1.0/task_servo_rate;
    uint        model_size                  = 25;
    double      tau                         = MIN_TAU;

    double      tau_learn;
    Trajectory  critical_states_learn(2);

    uint        formula_type                = _SCHAAL_DMP_;
    uint        canonical_order             = 2;        // default is using 2nd order canonical system
    uint        learning_method             = _SCHAAL_LWR_METHOD_;
    char        loa_data_dir_path[1000];
    get_data_path(loa_data_dir_path,"/dmp_coupling/learn_obs_avoid/learning_algo_verification/");
    char        dmp_plot_dir_path[1000];
    get_plot_path(dmp_plot_dir_path,"/dmp_coupling/learn_obs_avoid/learning_algo_verification/dmp/");
    char        loa_plot_dir_path[1000];
    get_plot_path(loa_plot_dir_path,"/dmp_coupling/learn_obs_avoid/learning_algo_verification/loa/");
    char        var_input_file_path[1000]   = "";
    char        var_output_file_path[1000]  = "";
    char        rt_err_file_path[1000];
    get_rt_errors_path(rt_err_file_path,"/rt_err.txt");

    uint        AF_H14_feature_method       = _AF_H14_;

    double      regularization_const        = 5.0;

    VectorN		start_position;
    VectorN		goal_position;

    Vector3     obsctr_cart_position_global;
    double      obs_sphere_radius           = 0.05;
    Matrix4x4   ctraj_hmg_transform_global_to_local_matrix;

    static  struct  option  long_options[]              = {
        {"formula_type",            required_argument, 0,  'f' },
        {"canonical_order",         required_argument, 0,  'c' },
        {"learning_method",         required_argument, 0,  'm' },
        {"loa_plot_dir_path",       required_argument, 0,  'o' },
        {"rt_err_file_path",        required_argument, 0,  'e' },
        {0,                         0,                 0,  0   }
    };

    int         opt                         = 0;
    int         long_index                  = 0;
    while ((opt = getopt_long( argc, argv, "f:c:m:o:e:", long_options, &long_index )) != -1)
    {
        switch (opt)
        {
            case 'f'    : formula_type                  = atoi(optarg);
                break;
            case 'c'    : canonical_order               = atoi(optarg);
                break;
            case 'm'    : learning_method               = atoi(optarg);
                break;
            case 'o'    : strcpy(loa_plot_dir_path, optarg);
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
        (strcmp(loa_plot_dir_path, "") == 0) || (strcmp(rt_err_file_path, "") == 0))
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    // Initialize real-time assertor
    RealTimeAssertor                rt_assertor(rt_err_file_path);
    rt_assertor.clear_rt_err_file();

    DMPState                        ee_cart_state_global(3, &rt_assertor);
    DMPState                        obsctr_cart_state_global(3, &rt_assertor);

    ObstacleStates                  point_obstacles_cart_state_global(2);

    DataIO                          data_io_main(&rt_assertor);

    // Initialize tau system
    TauSystem                       tau_sys(MIN_TAU, &rt_assertor);

    // Initialize canonical system
    CanonicalSystemDiscrete         canonical_sys_discr(&tau_sys, &rt_assertor, canonical_order);

    TCLearnObsAvoidFeatureParameter tc_loa_feat_param(&rt_assertor,
                                                      &canonical_sys_discr,
                                                      _SPHERE_OBSTACLE_CENTER_RADIUS_FILE_FORMAT_,
                                                      AF_H14_feature_method,
                                                      3,3,
                                                      (6.0/M_PI),(14.0/M_PI),
                                                      10.0, 30.0,
                                                      1,
                                                      7.0, 8.0);

    TransformCouplingLearnObsAvoid	transform_coupling_learn_obs_avoid(&tc_loa_feat_param,
                                                                       &tau_sys,
                                                                       &ee_cart_state_global,
                                                                       &point_obstacles_cart_state_global,
                                                                       &ctraj_hmg_transform_global_to_local_matrix,
                                                                       &rt_assertor,
                                                                       false,
                                                                       true,
                                                                       loa_plot_dir_path);

    std::vector<TransformCoupling*>	transform_couplers(1);

    // Initialize Cartesian Coordinate DMP
    CartesianCoordDMP               cart_dmp(&canonical_sys_discr, model_size,
                                             formula_type, learning_method, &rt_assertor,
                                             _GSUTANTO_LOCAL_COORD_FRAME_, "", &transform_couplers);

    if (rt_assert_main(cart_dmp.learn(get_data_path("/dmp_coupling/learn_obs_avoid/learning_algo_verification/synthetic_data/baseline/endeff_trajs/").c_str(),
                                      task_servo_rate, &tau_learn, &critical_states_learn)) == false)
    {
        return (-1);
    }

    tau                             = tau_learn;
    if (rt_assert_main(data_io_main.writeMatrixToFile(loa_plot_dir_path,
                                                      "tau.txt", tau)) == false)
    {
        return (-1);
    }

    // Load synthetic obstacle avoidance weights
    if (rt_assert_main(transform_coupling_learn_obs_avoid.loadWeights(loa_data_dir_path,
                                                                      "loa_synthetic_weights.txt")) == false)
    {
        return (-1);
    }

    // DMP parameters
    MatrixNxMPtr                    f_weights_baseline;
    if (rt_assert_main(allocateMemoryIfNonRealTime(is_real_time, f_weights_baseline,
                                                   3, model_size)) == false)
    {
        return (-1);
    }
    f_weights_baseline->resize(3, model_size);
    VectorN                         f_A_learn_baseline(3);

    if (rt_assert_main(cart_dmp.getParams(*f_weights_baseline, f_A_learn_baseline)) == false)
    {
        return (-1);
    }

    start_position					= cart_dmp.getMeanStartPosition();
    goal_position					= cart_dmp.getMeanGoalPosition();

    DMPState                        current_state(3, &rt_assertor);

    DMPUnrollInitParams             dmp_unroll_init_parameters(tau, critical_states_learn,
                                                               &rt_assertor);

    /****************** without obstacle avoidance (START) ******************/
    // NOT using obstacle avoidance coupling term:
    transform_couplers[0]			= NULL;
    if (rt_assert_main(cart_dmp.setParams(*f_weights_baseline, f_A_learn_baseline)) == false)
    {
        return (-1);
    }
    if (rt_assert_main(cart_dmp.start(dmp_unroll_init_parameters)) == false)
    {
        return (-1);
    }

    // Run DMP and print state values:
    if (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet()) == false)
    {
        return (-1);
    }
    for (uint i = 0; i < (round(tau*task_servo_rate) + 1); ++i)
    {
        double  time = 1.0 * (i*dt);

        // Get the next state of the Cartesian DMP
        if (rt_assert_main(cart_dmp.getNextState(dt, true, current_state)) == false)
        {
            return (-1);
        }

        // Log state trajectory:
        printf("%.05f %.05f %.05f %.05f\n", time, current_state.getX()[0], current_state.getX()[1], current_state.getX()[2]);
    }
    cart_dmp.stopCollectingTrajectoryDataSet();
    sprintf(var_output_file_path, "%s/baseline/", dmp_plot_dir_path);
    if (rt_assert_main(cart_dmp.saveTrajectoryDataSet(var_output_file_path)) == false)
    {
        return (-1);
    }
    /****************** without obstacle avoidance (END) ******************/

    // Load sphere obstacle parameters of the demonstration setting:
    sprintf(var_input_file_path, "%s/synthetic_data/1/", loa_data_dir_path);
    if (rt_assert_main(transform_coupling_learn_obs_avoid.readSphereObstacleParametersFromFile(var_input_file_path,
                                                                                               obsctr_cart_position_global,
                                                                                               obs_sphere_radius)) == false)
    {
        return (-1);
    }
    obsctr_cart_state_global        = DMPState(obsctr_cart_position_global, &rt_assertor);

    /****************** with synthetic obstacle avoidance (START) ******************/
    // Using obstacle avoidance coupling term:
    transform_couplers[0]			= &transform_coupling_learn_obs_avoid;

    if (rt_assert_main(cart_dmp.setParams(*f_weights_baseline, f_A_learn_baseline)) == false)
    {
        return (-1);
    }
    if (rt_assert_main(cart_dmp.start(dmp_unroll_init_parameters)) == false)
    {
        return (-1);
    }
    current_state                               = cart_dmp.getCurrentState();
    ctraj_hmg_transform_global_to_local_matrix	= cart_dmp.getHomogeneousTransformMatrixGlobalToLocal();

    // Run DMP and print state values:
    if (rt_assert_main(transform_coupling_learn_obs_avoid.startCollectingTrajectoryDataSet()) == false)
    {
        return (-1);
    }
    if (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet()) == false)
    {
        return (-1);
    }
    for (uint i = 0; i < (round(tau*task_servo_rate) + 1); ++i)
    {
        double  time = 1.0 * (i*dt);

        // IMPORTANT PART here:
        // Here ee_cart_state_global is updated, using the current (or real-time) DMP unrolled trajectory.
        // These variable will in turn be used for computing
        // the features of obstacle avoidance coupling term.
        ee_cart_state_global            = current_state;
        if (rt_assert_main(transform_coupling_learn_obs_avoid.computePointObstaclesCartStateGlobalFromStaticSphereObstacle(ee_cart_state_global,
                                                                                                                           obsctr_cart_state_global,
                                                                                                                           obs_sphere_radius,
                                                                                                                           point_obstacles_cart_state_global)) == false)
        {
            return (-1);
        }

        // Get the next state of the Cartesian DMP
        if (rt_assert_main(cart_dmp.getNextState(dt, true, current_state)) == false)
        {
            return (-1);
        }

        // Log state trajectory:
        printf("%.05f %.05f %.05f %.05f\n", time, current_state.getX()[0], current_state.getX()[1], current_state.getX()[2]);
    }
    cart_dmp.stopCollectingTrajectoryDataSet();
    sprintf(var_output_file_path, "%s/w_synthetic_obs_avoidance/", dmp_plot_dir_path);
    if (rt_assert_main(cart_dmp.saveTrajectoryDataSet(var_output_file_path)) == false)
    {
        return (-1);
    }
    transform_coupling_learn_obs_avoid.stopCollectingTrajectoryDataSet();
    if (rt_assert_main(transform_coupling_learn_obs_avoid.saveTrajectoryDataSet(var_output_file_path)) == false)
    {
        return (-1);
    }
    /****************** with synthetic obstacle avoidance (END) ******************/

    // Move synthetically-generated obstacle avoidance demonstration trajectory to the data directory:
    sprintf(var_input_file_path, "%s/w_synthetic_obs_avoidance/transform_sys_state_global_trajectory.txt", dmp_plot_dir_path);
    sprintf(var_output_file_path, "%s/synthetic_data/1/endeff_trajs/1.txt", loa_data_dir_path);
    if (rt_assert_main(rename( var_input_file_path, var_output_file_path ) == 0) == false)
    {
        return (-1);
    }

    // Learn from the synthetic obstacle avoidance trajectory:
    sprintf(var_input_file_path, "%s/synthetic_data/", loa_data_dir_path);
    if (rt_assert_main(transform_coupling_learn_obs_avoid.learnCouplingTerm(var_input_file_path,
                                                                            task_servo_rate,
                                                                            &cart_dmp,
                                                                            regularization_const,
                                                                            false)) == false)
    {
        return (-1);
    }

    // Load ARD-learned obstacle avoidance weights
    if (rt_assert_main(transform_coupling_learn_obs_avoid.loadWeights(loa_plot_dir_path,
                                                                      "learn_obs_avoid_weights_matrix_ARD.txt")) == false)
    {
        return (-1);
    }

    /****************** with ARD-learned obstacle avoidance (START) ******************/
    // Using obstacle avoidance coupling term:
    transform_couplers[0]			= &transform_coupling_learn_obs_avoid;

    if (rt_assert_main(cart_dmp.setParams(*f_weights_baseline, f_A_learn_baseline)) == false)
    {
        return (-1);
    }
    if (rt_assert_main(cart_dmp.start(dmp_unroll_init_parameters)) == false)
    {
        return (-1);
    }
    current_state                               = cart_dmp.getCurrentState();
    ctraj_hmg_transform_global_to_local_matrix	= cart_dmp.getHomogeneousTransformMatrixGlobalToLocal();

    // Run DMP and print state values:
    if (rt_assert_main(transform_coupling_learn_obs_avoid.startCollectingTrajectoryDataSet()) == false)
    {
        return (-1);
    }
    if (rt_assert_main(cart_dmp.startCollectingTrajectoryDataSet()) == false)
    {
        return (-1);
    }
    for (uint i = 0; i < (round(tau*task_servo_rate) + 1); ++i)
    {
        double  time = 1.0 * (i*dt);

        // IMPORTANT PART here:
        // Here ee_cart_state_global is updated, using the current (or real-time) DMP unrolled trajectory.
        // These variable will in turn be used for computing
        // the features of obstacle avoidance coupling term.
        ee_cart_state_global            = current_state;
        if (rt_assert_main(transform_coupling_learn_obs_avoid.computePointObstaclesCartStateGlobalFromStaticSphereObstacle(ee_cart_state_global,
                                                                                                                           obsctr_cart_state_global,
                                                                                                                           obs_sphere_radius,
                                                                                                                           point_obstacles_cart_state_global)) == false)
        {
            return (-1);
        }

        // Get the next state of the Cartesian DMP
        if (rt_assert_main(cart_dmp.getNextState(dt, true, current_state)) == false)
        {
            return (-1);
        }

        // Log state trajectory:
        printf("%.05f %.05f %.05f %.05f\n", time, current_state.getX()[0], current_state.getX()[1], current_state.getX()[2]);
    }
    cart_dmp.stopCollectingTrajectoryDataSet();
    sprintf(var_output_file_path, "%s/w_ARD_learned_obs_avoidance/", dmp_plot_dir_path);
    if (rt_assert_main(cart_dmp.saveTrajectoryDataSet(var_output_file_path)) == false)
    {
        return (-1);
    }
    transform_coupling_learn_obs_avoid.stopCollectingTrajectoryDataSet();
    if (rt_assert_main(transform_coupling_learn_obs_avoid.saveTrajectoryDataSet(var_output_file_path)) == false)
    {
        return (-1);
    }
    /****************** with ARD-learned obstacle avoidance (END) ******************/

    return 0;
}
