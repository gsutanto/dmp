#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>

#include "amd_clmc_dmp/utility/DefinitionsDerived.h"
#include "amd_clmc_dmp/dmp_state/DMPState.h"
#include "amd_clmc_dmp/dmp_param/TauSystem.h"
#include "amd_clmc_dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "amd_clmc_dmp/dmp_multi_dim/DMPDiscreteMultiDim.h"
#include "amd_clmc_dmp/neural_nets/FFNNFinalPhaseLWRLayerPerDims.h"
#include "amd_clmc_dmp/cart_dmp/cart_coord_dmp/CartesianCoordDMP.h"
#include "amd_clmc_dmp/cart_dmp/quat_dmp/QuaternionDMP.h"
#include "amd_clmc_dmp/dmp_coupling/learn_tactile_feedback/TransformCouplingLearnTactileFeedback.h"
#include "amd_clmc_dmp/paths.h"

using namespace dmp;

void print_usage()
{
    printf("Usage: amd_clmc_dmp_cart_dmp_nnlwr_fitted_ct_unroll_demo [-o output_plot_dir_path] [-e rt_err_file_path]\n");
}

int main(int argc, char** argv)
{
    bool    is_real_time            = false;

    uint    setting_no              = 5;
    uint    trial_no                = 3;

    double  task_servo_rate         = 300.0;    // demonstration/training data frequency
    double  dt                      = 1.0/task_servo_rate;
    uint    model_size              = 25;

    uint    formula_type            = _SCHAAL_DMP_;
    uint    canonical_order         = 2;        // default is using 2nd order canonical system
    uint    learning_method         = _SCHAAL_LWR_METHOD_;
    double  tau_reproduce           = 0.0;

    uint    N_prims                 = 3;
    uint    N_sense                 = 3;
    uint    N_points_ave_init_offset= 5;
    uint    N_act                   = 2;

    std::vector< uint > topology(4);
    topology[0]         = 45;
    topology[1]         = 100;
    topology[2]         = 25;
    topology[3]         = 1;

    char    output_plot_dir_path[1000];
    get_plot_path(output_plot_dir_path,"/dmp_coupling/learn_tactile_feedback/scraping/");
    char    rt_err_file_path[1000];
    get_rt_errors_path(rt_err_file_path,"/rt_err.txt");

    static  struct  option  long_options[]          = {
            {"output_plot_dir_path",required_argument, 0,  'o' },
            {"rt_err_file_path",    required_argument, 0,  'e' },
            {0,                     0,                 0,  0   }
    };

    int     opt                     = 0;
    int     long_index              = 0;
    while ((opt = getopt_long(argc, argv,"o:e:", long_options, &long_index )) != -1)
    {
        switch (opt)
        {
            case 'o'    : strcpy(output_plot_dir_path, optarg);
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
        (tau_reproduce < 0.0) || (strcmp(output_plot_dir_path, "") == 0) || (strcmp(rt_err_file_path, "") == 0))
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    std::vector<std::string>    sense_name(N_sense);
    sense_name[0]               = "R_LF";
    sense_name[1]               = "R_RF";
    sense_name[2]               = "proprio";

    std::vector<uint>           sense_dimensionality(N_sense);
    sense_dimensionality[0]     = 19;
    sense_dimensionality[1]     = 19;
    sense_dimensionality[2]     = 7;

    // start index for each sensing modality in the aggregated feature vector X:
    std::vector<uint>           sense_start_idx_in_aggreg_feat_vec(N_sense);
    for (uint s=0; s<N_sense; ++s)
    {
        if (s == 0)
        {
            sense_start_idx_in_aggreg_feat_vec[0]   = 0;
        }
        else
        {
            sense_start_idx_in_aggreg_feat_vec[s]   = sense_start_idx_in_aggreg_feat_vec[s-1] + sense_dimensionality[s-1];
        }
    }
    uint    N_total_sense_dimensionality    = sense_start_idx_in_aggreg_feat_vec[N_sense-1] + sense_dimensionality[N_sense-1];

    std::vector<std::string>    act_name(N_act);
    act_name[0]                 = "position";
    act_name[1]                 = "orientation";

    std::vector<uint>           act_dimensionality(N_act);
    act_dimensionality[0]       = 3;
    act_dimensionality[1]       = 3;

    // start index for each action modality in the aggregated coupling term vector Ct_vector:
    std::vector<uint>           act_start_idx_in_aggreg_ct_vec(N_act);

    for (uint a=0; a<N_act; ++a)
    {
        if (a == 0)
        {
            act_start_idx_in_aggreg_ct_vec[0]   = 0;
        }
        else
        {
            act_start_idx_in_aggreg_ct_vec[a]   = act_start_idx_in_aggreg_ct_vec[a-1] + act_dimensionality[a-1];
        }
    }
    uint    N_total_act_dimensionality  = act_start_idx_in_aggreg_ct_vec[N_act-1] + act_dimensionality[N_act-1];

    // Initialize real-time assertor
    RealTimeAssertor        rt_assertor(rt_err_file_path);
    rt_assertor.clear_rt_err_file();

    // Initialize tau system
    TauSystem               tau_sys(MIN_TAU, &rt_assertor);

    // Initialize canonical system
    CanonicalSystemDiscrete can_sys_discr(&tau_sys, &rt_assertor, canonical_order);

    // Helper DataIO
    DataIO                  data_io(&rt_assertor);

    // Sensory Primitives
    std::vector< boost::shared_ptr<DMPDiscreteMultiDim> >   dmp_discrete_sense(N_sense);
    std::vector< std::vector<MatrixNxM> >       dmpd_sense_weights(N_sense);
    std::vector< std::vector<VectorN> >         dmpd_sense_A_learn(N_sense);
    std::vector< std::vector<VectorN> >         dmpd_sense_start(N_sense);
    std::vector< std::vector<VectorN> >         dmpd_sense_goal(N_sense);
    std::vector< std::vector<double> >          dmpd_sense_tau(N_sense);
    // Current sense states:
    std::vector< boost::shared_ptr<DMPState> >  sense_state(N_sense);
    // The "actual" sensor traces buffers:
    std::vector< std::vector<MatrixXxXPtr> >    X_sensor_trace(N_sense);
    std::vector< std::vector<VectorN> >         X_sensor_offset(N_sense);
    // Sense unrolling parameters:
    std::vector< std::vector<Trajectory> >      dmpd_sense_unroll_critical_states(N_sense);

    // Unrolling properties
    std::vector<uint>                           unroll_traj_length(N_prims);
    std::vector<double>                         unroll_tau(N_prims);

    // Feature matrix:
//    std::vector<MatrixTxSBC>                    X(N_prims);

    // Basis functions vector:
    VectorM                                     func_approx_basis_functions(model_size);
    VectorNN_N                                  normalized_phase_PSI_mult_phase_V(model_size);

    // Coupling term prediction (result) matrix:
//    std::vector<MatrixTxSBC>                    Ct(N_prims);
    VectorNN_N                                  Ct_vector(N_total_act_dimensionality);

    double                                      canonical_multiplier;

    std::vector< boost::shared_ptr<FFNNFinalPhaseLWRLayerPerDims> > ffnn_phaselwr(N_prims);

    std::vector< boost::shared_ptr<TransformCouplingLearnTactileFeedback> > tc_lft(N_act);

    // Coupling Terms
    std::vector< std::vector<TransformCoupling*> >  transform_couplers(N_act);

    // Action/Movement Primitives
    std::vector< boost::shared_ptr<DMPDiscrete> >       dmp_discrete_act(N_act);
    dmp_discrete_act[0] = boost::make_shared<CartesianCoordDMP>(&can_sys_discr, model_size,
                                                                formula_type, learning_method,
                                                                &rt_assertor,
                                                                2, "",
                                                                &(transform_couplers[0]));
    dmp_discrete_act[1] = boost::make_shared<QuaternionDMP>(model_size, &can_sys_discr,
                                                            learning_method, &rt_assertor, "",
                                                            &(transform_couplers[1]));
    boost::shared_ptr<QuaternionDMP>            dmp_discrete_quat   = boost::dynamic_pointer_cast<QuaternionDMP> (dmp_discrete_act[1]);
    std::vector< std::vector<MatrixNxM> >       dmpd_act_weights(N_act);
    std::vector< std::vector<VectorN> >         dmpd_act_A_learn(N_act);
    std::vector< std::vector<VectorN> >         dmpd_act_start(N_act);
    std::vector< std::vector<VectorN> >         dmpd_act_goal(N_act);
    std::vector< std::vector<double> >          dmpd_act_tau(N_act);
    // Current act states:
    DMPState                                    cart_coord_state    = DMPState(act_dimensionality[0], &rt_assertor);
    QuaternionDMPState                          quat_state          = QuaternionDMPState(&rt_assertor);
    // The action traces/trajectory buffers:
    //std::vector< std::vector<MatrixXxXPtr> >    X_act_trace(N_act);
    // Action unrolling parameters:
    std::vector<Trajectory>                     dmpd_cart_coord_unroll_critical_states(N_prims);
    std::vector<QuaternionTrajectory>           dmpd_quat_unroll_critical_states(N_prims);

    // Feature matrix:
    std::vector<MatrixTxSBC>                    pose_traj_unroll(N_prims);

    std::string prims_param_root_dir_path   = get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/learned_prims_params/");
    char    var_params_dir_path[1000]       = "";
    std::string trial_data_root_dir_path    = get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/unroll_test_dataset/all_prims/");
    char    var_data_dir_path[1000]         = "";
    char    var_file_name[1000]             = "";

    for (uint m=0; m<N_sense; ++m)
    {
        // Initialize Sensory Primitives
        dmp_discrete_sense[m]   = boost::make_shared<DMPDiscreteMultiDim>(sense_dimensionality[m],
                                                                          model_size, &can_sys_discr,
                                                                          formula_type,
                                                                          learning_method,
                                                                          &rt_assertor, "");
        sense_state[m]          = boost::make_shared<DMPState>(sense_dimensionality[m], &rt_assertor);
        dmpd_sense_unroll_critical_states[m].resize(N_prims);
        dmpd_sense_weights[m].resize(N_prims);
        dmpd_sense_A_learn[m].resize(N_prims);
        dmpd_sense_start[m].resize(N_prims);
        dmpd_sense_goal[m].resize(N_prims);
        dmpd_sense_tau[m].resize(N_prims);

        // Turn-off all scaling in Sensory Primitives:
        if (rt_assert_main(dmp_discrete_sense[m]->setScalingUsage(std::vector<bool>(sense_dimensionality[m], false))) == false)
        {
            printf("Failed setting scaling usage for sensing modality %s\n", sense_name[m].c_str());
            return (-1);
        }

        X_sensor_trace[m].resize(N_prims);
        X_sensor_offset[m].resize(N_prims);

        for (uint np=0; np<N_prims; ++np)
        {
            sprintf(var_params_dir_path, "%s/sense_%s/prim%u/", prims_param_root_dir_path.c_str(), sense_name[m].c_str(), np+1);
            // Load the Sensory Primitives' parameters:
            if (rt_assert_main(dmp_discrete_sense[m]->loadParams(var_params_dir_path,
                                                                "w", "A_learn", "start", "goal", "tau")) == false)
            {
                printf("Failed loading parameters for sensing modality %s, primitive #%u\n", sense_name[m].c_str(), np+1);
                return (-1);
            }

            // Store the parameters on buffers:
            if (rt_assert_main(dmp_discrete_sense[m]->getParams(dmpd_sense_weights[m][np],
                                                                dmpd_sense_A_learn[m][np])) == false)
            {
                printf("Failed getting weights or A_learn parameters for sensing modality %s, primitive #%u\n", sense_name[m].c_str(), np+1);
                return (-1);
            }
            dmpd_sense_start[m][np] = dmp_discrete_sense[m]->getMeanStartPosition();
            dmpd_sense_goal[m][np]  = dmp_discrete_sense[m]->getMeanGoalPosition();
            dmpd_sense_tau[m][np]   = dmp_discrete_sense[m]->getMeanTau();

            sprintf(var_data_dir_path, "%s/setting_%u_trial_%u/prim%u/", trial_data_root_dir_path.c_str(), setting_no, trial_no, np+1);
            // Load the "actual" sensor traces:
            std::string     file_name   = "sense_" + sense_name[m];
            if (rt_assert_main(data_io.readMatrixFromFile(var_data_dir_path, file_name.c_str(), X_sensor_trace[m][np])) == false)
            {
                printf("Failed reading actual sensor traces for sensing modality %s, primitive #%u\n", sense_name[m].c_str(), np+1);
                return (-1);
            }
            if (rt_assert_main(X_sensor_trace[m][np]->cols() == sense_dimensionality[m]) == false)
            {
                printf("Failed reading actual sensor traces for sensing modality %s, primitive #%u\n", sense_name[m].c_str(), np+1);
                return (-1);
            }

            // initial sensor offset averaging
            if (m < 2)  // BioTac electrodes
            {
                if (np == 0)
                {
                    X_sensor_offset[m][np]  = ZeroVectorN(sense_dimensionality[m]);
                    for (uint t=0; t<N_points_ave_init_offset; ++t)
                    {
                        X_sensor_offset[m][np]  = X_sensor_offset[m][np] + (1.0 * X_sensor_trace[m][np]->block(t, 0, 1, sense_dimensionality[m]).transpose());
                    }
                    X_sensor_offset[m][np]  = (1.0/N_points_ave_init_offset) * X_sensor_offset[m][np];
                }
                else
                {
                    X_sensor_offset[m][np]  = X_sensor_offset[m][0];
                }
            }
            else    // Joint/Proprioception (no offset)
            {
                X_sensor_offset[m][np]  = ZeroVectorN(sense_dimensionality[m]);
            }

            if (m == 0)
            {
                unroll_traj_length[np]  = X_sensor_trace[m][np]->rows();
                unroll_tau[np]          = (unroll_traj_length[np] - 1) * dt;
            }
            else
            {
                if (rt_assert_main(unroll_traj_length[np] == X_sensor_trace[m][np]->rows()) == false)
                {
                    printf("Unroll trajectory length is INCONSISTENT on sensing modality %s!!!\n", sense_name[m].c_str());
                    return (-1);
                }
            }

            dmpd_sense_unroll_critical_states[m][np].resize(2);
        }
    }

    VectorNN_N  X_vector(N_total_sense_dimensionality);

    char nn_lwr_model_path[1000]    = "";
    for (uint np=0; np<N_prims; ++np)
    {
        // Initialize Neural Network-LWR (NN-LWR):
        ffnn_phaselwr[np]   = boost::make_shared<FFNNFinalPhaseLWRLayerPerDims>(N_total_act_dimensionality,
                                                                                topology,
                                                                                &rt_assertor);

        sprintf(nn_lwr_model_path, "%sprim%u/",
                get_data_path("/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/FFNNFinalPhaseLWRLayerPerDims/cpp_models/").c_str(),
                np+1);
        if (rt_assert_main(ffnn_phaselwr[np]->loadParams(nn_lwr_model_path, 0)) == false)
        {
            printf("Failed loading NN-LWR params for primitive #%u\n", np+1);
            return (-1);
        }
    }

    for (uint a=0; a<N_act; ++a)
    {
        dmpd_act_weights[a].resize(N_prims);
        dmpd_act_A_learn[a].resize(N_prims);
        dmpd_act_start[a].resize(N_prims);
        dmpd_act_goal[a].resize(N_prims);
        dmpd_act_tau[a].resize(N_prims);

        // Turn-off all scaling in Movement Primitives:
        if (rt_assert_main(dmp_discrete_act[a]->setScalingUsage(std::vector<bool>(3, false))) == false)
        {
            printf("Failed setting scaling usage for %s primitive.\n", act_name[a].c_str());
            return (-1);
        }

        //X_act_trace[a].resize(N_prims);

        for (uint np=0; np<N_prims; ++np)
        {
            sprintf(var_params_dir_path, "%s/%s/prim%u/", prims_param_root_dir_path.c_str(), act_name[a].c_str(), np+1);
            // Load the Action Primitives' parameters:
            if (a == 0)
            {
                if (rt_assert_main(dmp_discrete_act[a]->loadParams(var_params_dir_path,
                                                                   "w", "A_learn", "start_global", "goal_global", "tau")) == false)
                {
                    printf("Failed loading parameters for action modality %s, primitive #%u\n", act_name[a].c_str(), np+1);
                    return (-1);
                }
            }
            else // if (a == 1)
            {
                if (rt_assert_main(dmp_discrete_act[a]->loadParams(var_params_dir_path,
                                                                   "w", "A_learn", "start", "goal", "tau")) == false)
                {
                    printf("Failed loading parameters for action modality %s, primitive #%u\n", act_name[a].c_str(), np+1);
                    return (-1);
                }
            }

            // Store the parameters on buffers:
            if (rt_assert_main(dmp_discrete_act[a]->getParams(dmpd_act_weights[a][np],
                                                              dmpd_act_A_learn[a][np])) == false)
            {
                printf("Failed getting weights or A_learn parameters for action modality %s, primitive #%u\n", act_name[a].c_str(), np+1);
                return (-1);
            }
            dmpd_act_start[a][np]   = dmp_discrete_act[a]->getMeanStartPosition();
            dmpd_act_goal[a][np]    = dmp_discrete_act[a]->getMeanGoalPosition();
            dmpd_act_tau[a][np]     = dmp_discrete_act[a]->getMeanTau();
        }

        // Initialize Transform Coupling Learn Tactile Feedback:
        tc_lft[a]   = boost::make_shared<TransformCouplingLearnTactileFeedback>(act_dimensionality[a],
                                                                                act_start_idx_in_aggreg_ct_vec[a],
                                                                                topology[0],
                                                                                model_size,
                                                                                N_total_act_dimensionality,
                                                                                &X_vector,
                                                                                &normalized_phase_PSI_mult_phase_V,
                                                                                &Ct_vector,
                                                                                &rt_assertor);
        transform_couplers[a].resize(1);
        transform_couplers[a][0]    = tc_lft[a].get();
    }

    for (uint np=0; np<N_prims; ++np)
    {
        dmpd_cart_coord_unroll_critical_states[np].resize(2);
        dmpd_quat_unroll_critical_states[np].resize(2);

//        X[np].resize(unroll_traj_length[np], N_total_sense_dimensionality);
//        Ct[np].resize(unroll_traj_length[np], N_total_act_dimensionality);
        pose_traj_unroll[np].resize(unroll_traj_length[np], 7);
    }

    /**** Codes below this should work in real-time control ****/

    /**** Reproduce ****/
    for (uint np=0; np<N_prims; ++np)
    {
        // Starting the sensory primitives:
        for (uint m=0; m<N_sense; ++m)
        {
            if (rt_assert_main(dmp_discrete_sense[m]->setParams(dmpd_sense_weights[m][np],
                                                                dmpd_sense_A_learn[m][np])) == false)
            {
                printf("Failed setting weights or A_learn parameters for sensing modality %s, primitive #%u\n", sense_name[m].c_str(), np+1);
                return (-1);
            }

            dmpd_sense_unroll_critical_states[m][np][0] = DMPState(dmpd_sense_start[m][np], &rt_assertor);
            dmpd_sense_unroll_critical_states[m][np][1] = DMPState(dmpd_sense_goal[m][np], &rt_assertor);

            if (rt_assert_main(dmp_discrete_sense[m]->start(dmpd_sense_unroll_critical_states[m][np], unroll_tau[np])) == false)
            {
                printf("Failed starting sensory primitive for sensing modality %s, primitive #%u\n", sense_name[m].c_str(), np+1);
                return (-1);
            }
        }

        // Starting the action primitives:
        for (uint a=0; a<N_act; ++a)
        {
            if (rt_assert_main(dmp_discrete_act[a]->setParams(dmpd_act_weights[a][np],
                                                              dmpd_act_A_learn[a][np])) == false)
            {
                printf("Failed setting weights or A_learn parameters for action modality %s, primitive #%u\n", act_name[a].c_str(), np+1);
                return (-1);
            }

            if (np == 0)
            {
                if (a == 0)
                {
                    dmpd_cart_coord_unroll_critical_states[np][0]   = DMPState(dmpd_act_start[a][np], &rt_assertor);
                    dmpd_cart_coord_unroll_critical_states[np][1]   = DMPState(dmpd_act_goal[a][np], &rt_assertor);
                }
                else // if (a == 1)
                {
                    dmpd_quat_unroll_critical_states[np][0]         = QuaternionDMPState(dmpd_act_start[a][np], &rt_assertor);
                    dmpd_quat_unroll_critical_states[np][1]         = QuaternionDMPState(dmpd_act_goal[a][np], &rt_assertor);
                }
            }
            else
            {
                if (a == 0)
                {
                    dmpd_cart_coord_unroll_critical_states[np][0]   = cart_coord_state;
                    dmpd_cart_coord_unroll_critical_states[np][1]   = DMPState(dmpd_act_goal[a][np], &rt_assertor);
                }
                else // if (a == 1)
                {
                    dmpd_quat_unroll_critical_states[np][0]         = quat_state;
                    dmpd_quat_unroll_critical_states[np][1]         = QuaternionDMPState(dmpd_act_goal[a][np], &rt_assertor);
                }
            }

            if (a == 0)
            {
                if (rt_assert_main(dmp_discrete_act[a]->start(dmpd_cart_coord_unroll_critical_states[np], unroll_tau[np])) == false)
                {
                    printf("Failed starting sensory primitive for action modality %s, primitive #%u\n", act_name[a].c_str(), np+1);
                    return (-1);
                }
            }
            else // if (a == 1)
            {
                if (rt_assert_main(dmp_discrete_quat->startQuaternionDMP(dmpd_quat_unroll_critical_states[np], unroll_tau[np])) == false)
                {
                    printf("Failed starting sensory primitive for action modality %s, primitive #%u\n", act_name[a].c_str(), np+1);
                    return (-1);
                }
            }

            // Select the activated coupling term's neural network:
            (*(tc_lft[a])).ffnn_lwr_per_dims    = ffnn_phaselwr[np].get();
        }

        // Run sensory primitives:
        for(uint t = 0; t < unroll_traj_length[np]; ++t)
        {
            for (uint m=0; m<N_sense; ++m)
            {
                if (t == 0)
                {
                    *(sense_state[m])   = dmp_discrete_sense[m]->getCurrentState();
                }
                else
                {
                    // Get the next state of the sensory primitives
                    if (rt_assert_main(dmp_discrete_sense[m]->getNextState(dt, false, *(sense_state[m]),
                                                                           NULL, NULL, NULL,
                                                                           &func_approx_basis_functions)) == false)
                    {
                        printf("Failed sensory primitive unrolling at time t=%u for sensing modality %s, primitive #%u\n", t,
                               sense_name[m].c_str(), np+1);
                        return (-1);
                    }
                }
                X_vector.block(sense_start_idx_in_aggreg_feat_vec[m], 0, sense_dimensionality[m], 1)
                        = (X_sensor_trace[m][np]->block(t, 0, 1, sense_dimensionality[m]).transpose()
                           - X_sensor_offset[m][np].block(0, 0, sense_dimensionality[m], 1))
                          - sense_state[m]->getX(); // DeltaX = Xsense - Xnominal;
            }

            // Get the next state of the canonical system
            if (t > 0)
            {
                if (rt_assert_main(can_sys_discr.updateCanonicalState(dt)) == false)
                {
                    printf("Failed updating canonical state!!!\n");
                    return (-1);
                }
            }

            canonical_multiplier    = can_sys_discr.getCanonicalMultiplier();
            double sum_psi          = (func_approx_basis_functions.colwise().sum())(0,0) + (model_size * 1.e-10);
            normalized_phase_PSI_mult_phase_V   = func_approx_basis_functions * (canonical_multiplier/sum_psi);

            for (uint a=0; a<N_act; ++a)
            {
                // Get the next state of the action primitives
                if (a == 0)
                {
                    if (rt_assert_main(dmp_discrete_act[a]->getNextState(dt, false, cart_coord_state)) == false)
                    {
                        printf("Failed action primitive unrolling at time t=%u for action modality %s, primitive #%u\n", t,
                               act_name[a].c_str(), np+1);
                        return (-1);
                    }
                }
                else // if (a == 1)
                {
                    if (rt_assert_main(dmp_discrete_quat->getNextQuaternionState(dt, false, quat_state)) == false)
                    {
                        printf("Failed action primitive unrolling at time t=%u for action modality %s, primitive #%u\n", t,
                               act_name[a].c_str(), np+1);
                        return (-1);
                    }
                }
            }
            pose_traj_unroll[np].block(t, 0, 1, 3)  = cart_coord_state.getX().block(0,0,3,1).transpose();
            pose_traj_unroll[np].block(t, 3, 1, 4)  = quat_state.getQ().block(0,0,4,1).transpose();
            
            // actually the following computePrediction() is unnecessary,
            // since inside the coupling term getValue(), this is called automatically;
            // the following is called only for logging purpose:
            if (rt_assert_main(ffnn_phaselwr[np]->computePrediction(X_vector,
                                                                    normalized_phase_PSI_mult_phase_V,
                                                                    Ct_vector)) == false)
            {
                printf("Failed prediction at time t=%u for primitive #%u\n", t, np+1);
                return (-1);
            }

            // data logging for verification:
//            X[np].block(t, 0, 1, N_total_sense_dimensionality)  = X_vector.transpose();
//            Ct[np].block(t, 0, 1, N_total_act_dimensionality)   = Ct_vector.transpose();
        }

//        sprintf(var_file_name, "X_prim%u", np+1);
//        if (rt_assert_main(data_io.writeMatrixToFile(output_plot_dir_path, var_file_name, X[np])) == false)
//        {
//            printf("Failed to write feature matrix X of primitive #%u\n", np+1);
//            return (-1);
//        }

//        sprintf(var_file_name, "Ct_prim%u", np+1);
//        if (rt_assert_main(data_io.writeMatrixToFile(output_plot_dir_path, var_file_name, Ct[np])) == false)
//        {
//            printf("Failed to write predicted coupling term matrix Ct of primitive #%u\n", np+1);
//            return (-1);
//        }

        sprintf(var_file_name, "test_cpp_coupled_cart_dmp_unroll_test_prim%u", np+1);
        if (rt_assert_main(data_io.writeMatrixToFile(output_plot_dir_path, var_file_name, pose_traj_unroll[np])) == false)
        {
            printf("Failed to write coupled trajectory unrolling of primitive #%u\n", np+1);
            return (-1);
        }
    }

    return 0;
}
