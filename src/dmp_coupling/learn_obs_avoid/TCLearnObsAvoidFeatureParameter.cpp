#include "amd_clmc_dmp/dmp_coupling/learn_obs_avoid/TCLearnObsAvoidFeatureParameter.h"

namespace dmp
{

/**
 * NON-REAL-TIME!!!
 */
TCLearnObsAvoidFeatureParameter::TCLearnObsAvoidFeatureParameter():
    rt_assertor(NULL),
    canonical_sys_discrete(NULL),
    feature_vector_size(0),
    data_file_format(0),
    is_using_AF_H14_feature(false),
    AF_H14_feature_method(0),
    AF_H14_feature_matrix_start_idx(0),
    AF_H14_feature_matrix_end_idx(0),
    AF_H14_feature_matrix_rows(0),
    AF_H14_num_indep_feat_obs_points(0),
    AF_H14_N_beta_phi1_phi2_grid(0),
    AF_H14_N_k_phi1_phi2_grid(0),
    AF_H14_N_k_phi3_grid(0),
    is_using_PF_DYN2_feature(false),
    PF_DYN2_feature_method(0),
    PF_DYN2_feature_matrix_start_idx(0),
    PF_DYN2_feature_matrix_end_idx(0),
    PF_DYN2_feature_matrix_rows(0),
    PF_DYN2_N_beta_grid(0),
    PF_DYN2_N_k_grid(0),
    is_using_KGF_feature(false),
    KGF_feature_method(0),
    KGF_feature_matrix_start_idx(0),
    KGF_feature_matrix_end_idx(0),
    KGF_feature_matrix_rows(0),
    KGF_N_beta_grid(0),
    KGF_N_k_grid(0),
    KGF_N_s_grid(0),
    is_using_NN(false),
    NN_feature_method(0),
    NN_feature_vector_rows(0),
    NN_N_input(0),
    NN_N_hidden_layer_1(0),
    NN_N_hidden_layer_2(0),
    NN_N_output(0),
    pmnn(NULL),
    pmnn_input_vector(NULL),
    pmnn_phase_kernel_modulation(NULL),
    pmnn_output_vector(NULL)
{}

/**
 * NON-REAL-TIME!!!
 */
TCLearnObsAvoidFeatureParameter::TCLearnObsAvoidFeatureParameter(RealTimeAssertor* real_time_assertor,
                                                                 CanonicalSystemDiscrete* canonical_system_discrete,
                                                                 uint input_data_file_format,
                                                                 uint AF_H14_feat_method,
                                                                 uint Num_AF_H14_beta_phi1_phi2_grid,
                                                                 uint Num_AF_H14_k_phi1_phi2_grid,
                                                                 double init_AF_H14_beta_phi1_phi2_low, double init_AF_H14_beta_phi1_phi2_high,
                                                                 double init_AF_H14_k_phi1_phi2_low, double init_AF_H14_k_phi1_phi2_high,
                                                                 uint Num_AF_H14_k_phi3_grid,
                                                                 double init_AF_H14_k_phi3_low, double init_AF_H14_k_phi3_high,
                                                                 uint init_distance_kernel_scale_grid_mode,
                                                                 uint PF_DYN2_feat_method,
                                                                 uint Num_PF_DYN2_beta_grid,
                                                                 uint Num_PF_DYN2_k_grid,
                                                                 double init_PF_DYN2_beta_low, double init_PF_DYN2_beta_high,
                                                                 double init_PF_DYN2_k_low, double init_PF_DYN2_k_high,
                                                                 uint KGF_feat_method,
                                                                 uint Num_KGF_beta_row_col_grid,
                                                                 uint Num_KGF_k_row_col_grid,
                                                                 uint Num_KGF_s_grid,
                                                                 double init_KGF_beta_low, double init_KGF_beta_high,
                                                                 double init_KGF_k_low, double init_KGF_k_high,
                                                                 double init_KGF_s_low, double init_KGF_s_high,
                                                                 uint NN_feat_method,
                                                                 uint Num_NN_input,
                                                                 uint Num_NN_hidden_layer_1,
                                                                 uint Num_NN_hidden_layer_2,
                                                                 uint Num_NN_output,
                                                                 const char* NN_params_directory_path,
                                                                 VectorNN_N* pmnn_input_vector_ptr,
                                                                 VectorNN_N* pmnn_phase_kernel_modulation_ptr,
                                                                 VectorNN_N* pmnn_output_vector_ptr):
    loa_data_io(LearnObsAvoidDataIO(real_time_assertor)),
    rt_assertor(real_time_assertor),
    canonical_sys_discrete(canonical_system_discrete),
    data_file_format(input_data_file_format),
    distance_kernel_scale_grid_mode(init_distance_kernel_scale_grid_mode),
    AF_H14_feature_method(AF_H14_feat_method),
    AF_H14_N_beta_phi1_phi2_grid(Num_AF_H14_beta_phi1_phi2_grid),
    AF_H14_N_k_phi1_phi2_grid(Num_AF_H14_k_phi1_phi2_grid),
    AF_H14_N_k_phi3_grid(Num_AF_H14_k_phi3_grid),
    AF_H14_beta_phi1_phi2_low(init_AF_H14_beta_phi1_phi2_low),
    AF_H14_beta_phi1_phi2_high(init_AF_H14_beta_phi1_phi2_high),
    AF_H14_k_phi1_phi2_low(init_AF_H14_k_phi1_phi2_low),
    AF_H14_k_phi1_phi2_high(init_AF_H14_k_phi1_phi2_high),
    AF_H14_k_phi3_low(init_AF_H14_k_phi3_low),
    AF_H14_k_phi3_high(init_AF_H14_k_phi3_high),
    PF_DYN2_feature_method(PF_DYN2_feat_method),
    PF_DYN2_N_beta_grid(Num_PF_DYN2_beta_grid),
    PF_DYN2_N_k_grid(Num_PF_DYN2_k_grid),
    PF_DYN2_beta_low(init_PF_DYN2_beta_low),
    PF_DYN2_beta_high(init_PF_DYN2_beta_high),
    PF_DYN2_k_low(init_PF_DYN2_k_low),
    PF_DYN2_k_high(init_PF_DYN2_k_high),
    KGF_feature_method(KGF_feat_method),
    KGF_N_beta_grid(Num_KGF_beta_row_col_grid),
    KGF_N_k_grid(Num_KGF_k_row_col_grid),
    KGF_N_s_grid(Num_KGF_s_grid),
    KGF_beta_low(init_KGF_beta_low),
    KGF_beta_high(init_KGF_beta_high),
    KGF_k_low(init_KGF_k_low),
    KGF_k_high(init_KGF_k_high),
    KGF_s_low(init_KGF_s_low),
    KGF_s_high(init_KGF_s_high),
    NN_feature_method(NN_feat_method),
    NN_N_input(Num_NN_input),
    NN_N_hidden_layer_1(Num_NN_hidden_layer_1),
    NN_N_hidden_layer_2(Num_NN_hidden_layer_2),
    NN_N_output(Num_NN_output),
    pmnn(NULL),
    pmnn_input_vector(pmnn_input_vector_ptr),
    pmnn_phase_kernel_modulation(pmnn_phase_kernel_modulation_ptr),
    pmnn_output_vector(pmnn_output_vector_ptr)
{
    bool is_real_time           = false;

    if (NN_feature_method == _NN_NO_USE_)
    {
        is_using_NN             = false;

        if ((AF_H14_feature_method > _AF_H14_NO_USE_) &&
            (AF_H14_feature_method <= _NUM_AF_H14_FEATURE_METHODS_))
        {
            is_using_AF_H14_feature         = true;

            AF_H14_feature_matrix_start_idx = 0;

            // compute AF_H14 feature parameters: beta and k for phi_1 and phi_2, k for phi_3
            AF_H14_N_beta_phi1_phi2_vector  = AF_H14_N_k_phi1_phi2_grid * AF_H14_N_beta_phi1_phi2_grid;
            AF_H14_N_k_phi1_phi2_vector     = AF_H14_N_k_phi1_phi2_grid * AF_H14_N_beta_phi1_phi2_grid;

            // compute feature size based on selected feature method:
            if ((AF_H14_feature_method == _AF_H14_) ||
                (AF_H14_feature_method == _AF_H14_NO_PHI3_))
            {
                AF_H14_num_indep_feat_obs_points   = 2;
            }
            else if ((AF_H14_feature_method == _AF_H14_NO_PHI3_SUM_OBS_POINTS_CONTRIBUTION_) ||
                     (AF_H14_feature_method == _AF_H14_SUM_OBS_POINTS_CONTRIBUTION_))
            {
                AF_H14_num_indep_feat_obs_points   = 1;
            }
            AF_H14_feature_matrix_rows             = (AF_H14_num_indep_feat_obs_points *
                                                      AF_H14_N_beta_phi1_phi2_grid *
                                                      AF_H14_N_k_phi1_phi2_grid);
            if ((AF_H14_feature_method == _AF_H14_) ||
                (AF_H14_feature_method == _AF_H14_SUM_OBS_POINTS_CONTRIBUTION_))
            {
                AF_H14_feature_matrix_rows          += AF_H14_N_k_phi3_grid;
            }
            else
            {
                AF_H14_N_k_phi3_grid                = 0;
            }
            AF_H14_feature_matrix_end_idx           = AF_H14_feature_matrix_start_idx +
                                                      AF_H14_feature_matrix_rows - 1;

            allocateMemoryIfNonRealTime(is_real_time, AF_H14_beta_phi1_phi2_grid,
                                        AF_H14_N_k_phi1_phi2_grid, AF_H14_N_beta_phi1_phi2_grid);
            allocateMemoryIfNonRealTime(is_real_time, AF_H14_k_phi1_phi2_grid,
                                        AF_H14_N_k_phi1_phi2_grid, AF_H14_N_beta_phi1_phi2_grid);
            allocateMemoryIfNonRealTime(is_real_time, AF_H14_beta_phi1_phi2_vector,
                                        AF_H14_N_beta_phi1_phi2_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, AF_H14_k_phi1_phi2_vector,
                                        AF_H14_N_k_phi1_phi2_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point,
                                        3, AF_H14_N_beta_phi1_phi2_vector);
            if (AF_H14_N_k_phi3_grid > 0)
            {
                allocateMemoryIfNonRealTime(is_real_time, AF_H14_k_phi3_vector,
                                            AF_H14_N_k_phi3_grid, 1);
                allocateMemoryIfNonRealTime(is_real_time, AF_H14_loa_feature_matrix_phi3_per_obs_point,
                                            3, AF_H14_N_k_phi3_grid);
            }

            if (AF_H14_N_beta_phi1_phi2_grid > 1)
            {
                *AF_H14_beta_phi1_phi2_grid = Eigen::RowVectorXd::LinSpaced(AF_H14_N_beta_phi1_phi2_grid,
                                                                            AF_H14_beta_phi1_phi2_low,
                                                                            AF_H14_beta_phi1_phi2_high).replicate(AF_H14_N_k_phi1_phi2_grid,
                                                                                                                  1);
            }
            else
            {
                *AF_H14_beta_phi1_phi2_grid = AF_H14_beta_phi1_phi2_low * Eigen::VectorXd::Ones(AF_H14_N_k_phi1_phi2_grid);
            }
            *AF_H14_beta_phi1_phi2_vector   = Eigen::Map<Eigen::MatrixXd>(AF_H14_beta_phi1_phi2_grid->data(),
                                                                          AF_H14_N_beta_phi1_phi2_vector,1);

            if (AF_H14_N_k_phi1_phi2_grid > 1)
            {
                *AF_H14_k_phi1_phi2_grid    = Eigen::VectorXd::LinSpaced(AF_H14_N_k_phi1_phi2_grid,
                                                                         AF_H14_k_phi1_phi2_low,
                                                                         AF_H14_k_phi1_phi2_high).replicate(1,
                                                                                                            AF_H14_N_beta_phi1_phi2_grid);
            }
            else
            {
                *AF_H14_k_phi1_phi2_grid    = AF_H14_k_phi1_phi2_high * Eigen::RowVectorXd::Ones(AF_H14_N_beta_phi1_phi2_grid);
            }
            if (distance_kernel_scale_grid_mode == _QUADRATIC_)
            {
                *AF_H14_k_phi1_phi2_grid    = AF_H14_k_phi1_phi2_grid->array().square();
            }
            else if (distance_kernel_scale_grid_mode == _INVERSE_QUADRATIC_)
            {
                *AF_H14_k_phi1_phi2_grid    = AF_H14_k_phi1_phi2_grid->array().inverse().array().square();
            }
            *AF_H14_k_phi1_phi2_vector      = Eigen::Map<Eigen::MatrixXd>(AF_H14_k_phi1_phi2_grid->data(),
                                                                          AF_H14_N_k_phi1_phi2_vector,1);

            if (AF_H14_N_k_phi3_grid >= 1)
            {
                if (AF_H14_N_k_phi3_grid > 1)
                {
                    *AF_H14_k_phi3_vector       = Eigen::VectorXd::LinSpaced(AF_H14_N_k_phi3_grid,
                                                                             AF_H14_k_phi3_low,
                                                                             AF_H14_k_phi3_high);
                }
                else // if (AF_H14_N_k_phi3_grid == 1)
                {
                    (*AF_H14_k_phi3_vector)(0)  = AF_H14_k_phi3_high;
                }
                if (distance_kernel_scale_grid_mode == _QUADRATIC_)
                {
                    *AF_H14_k_phi3_vector       = AF_H14_k_phi3_vector->array().square();
                }
                else if (distance_kernel_scale_grid_mode == _INVERSE_QUADRATIC_)
                {
                    *AF_H14_k_phi3_vector       = AF_H14_k_phi3_vector->array().inverse().array().square();
                }
            }
        }
        else // if NOT using AF_H14 features
        {
            is_using_AF_H14_feature         = false;

            AF_H14_feature_matrix_start_idx = -1;
            AF_H14_feature_matrix_end_idx   = -1;
            AF_H14_feature_matrix_rows      = 0;
            AF_H14_num_indep_feat_obs_points= 0;

            AF_H14_N_beta_phi1_phi2_grid    = 0;
            AF_H14_N_k_phi1_phi2_grid       = 0;
            AF_H14_N_k_phi3_grid            = 0;
            AF_H14_N_beta_phi1_phi2_vector  = 0;
            AF_H14_N_k_phi1_phi2_vector     = 0;
            AF_H14_beta_phi1_phi2_low       = 0;
            AF_H14_beta_phi1_phi2_high      = 0;
            AF_H14_k_phi1_phi2_low          = 0;
            AF_H14_k_phi1_phi2_high         = 0;
            AF_H14_k_phi3_low               = 0;
            AF_H14_k_phi3_high              = 0;
        }

        if ((PF_DYN2_feature_method > _PF_DYN2_NO_USE_) &&
            (PF_DYN2_feature_method <= _NUM_PF_DYN2_FEATURE_METHODS_))
        {
            is_using_PF_DYN2_feature        = true;

            PF_DYN2_feature_matrix_start_idx= AF_H14_feature_matrix_end_idx + 1;

            // compute PF_DYN2 feature parameters: beta and k
            PF_DYN2_N_beta_vector           = PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid;
            PF_DYN2_N_k_vector              = PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid;

            // compute feature size based on selected feature method:
            PF_DYN2_feature_matrix_rows     = (PF_DYN2_N_beta_grid * PF_DYN2_N_k_grid);
            PF_DYN2_feature_matrix_end_idx  = PF_DYN2_feature_matrix_start_idx +
                                              PF_DYN2_feature_matrix_rows - 1;

            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_beta_grid,
                                        PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_beta_D_grid,
                                        PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_k_grid,
                                        PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_k_D_grid,
                                        PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_beta_vector,
                                        PF_DYN2_N_beta_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_beta_D_vector,
                                        PF_DYN2_N_beta_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_k_vector,
                                        PF_DYN2_N_k_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_k_D_vector,
                                        PF_DYN2_N_k_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, PF_DYN2_loa_feature_matrix_per_obs_point,
                                        3, PF_DYN2_N_beta_vector);

            if (PF_DYN2_N_beta_grid > 1)
            {
                *PF_DYN2_beta_grid  = Eigen::RowVectorXd::LinSpaced(PF_DYN2_N_beta_grid,
                                                                    PF_DYN2_beta_low,
                                                                    PF_DYN2_beta_high).replicate(PF_DYN2_N_k_grid,
                                                                                                 1);
                for (uint j=0; j<PF_DYN2_N_beta_grid; j++)
                {
                    if (j < (PF_DYN2_N_beta_grid-1))
                    {
                        (*PF_DYN2_beta_D_grid)(0,j) = (*PF_DYN2_beta_grid)(0,j+1) - (*PF_DYN2_beta_grid)(0,j);
                    }
                    else if (j == (PF_DYN2_N_beta_grid-1))
                    {
                        (*PF_DYN2_beta_D_grid)(0,j) = (*PF_DYN2_beta_grid)(0,j) - (*PF_DYN2_beta_grid)(0,j-1);
                    }
                }
            }
            else
            {
                *PF_DYN2_beta_grid          = PF_DYN2_beta_low * Eigen::VectorXd::Ones(PF_DYN2_N_k_grid);
                (*PF_DYN2_beta_D_grid)(0,0) = DEFAULT_KERNEL_SPREAD;
            }
            if (PF_DYN2_N_k_grid > 1)
            {
                PF_DYN2_beta_D_grid->block(1,0,PF_DYN2_N_k_grid-1,PF_DYN2_N_beta_grid)  = PF_DYN2_beta_D_grid->block(0,0,1,PF_DYN2_N_beta_grid).replicate(PF_DYN2_N_k_grid-1,
                                                                                                                                                          1);
            }
            (*PF_DYN2_beta_D_grid)  = (0.55 * PF_DYN2_beta_D_grid->array()).array().square().cwiseInverse();
            *PF_DYN2_beta_vector    = Eigen::Map<Eigen::MatrixXd>(PF_DYN2_beta_grid->data(),
                                                                  PF_DYN2_N_beta_vector,1);
            *PF_DYN2_beta_D_vector  = Eigen::Map<Eigen::MatrixXd>(PF_DYN2_beta_D_grid->data(),
                                                                  PF_DYN2_N_beta_vector,1);

            if (PF_DYN2_N_k_grid > 1)
            {
                *PF_DYN2_k_grid     = Eigen::VectorXd::LinSpaced(PF_DYN2_N_k_grid,
                                                                 PF_DYN2_k_low,
                                                                 PF_DYN2_k_high).replicate(1,
                                                                                           PF_DYN2_N_beta_grid);
                for (uint i=0; i<PF_DYN2_N_k_grid; i++)
                {
                    if (i < (PF_DYN2_N_k_grid-1))
                    {
                        (*PF_DYN2_k_D_grid)(i,0)    = (*PF_DYN2_k_grid)(i+1,0) - (*PF_DYN2_k_grid)(i,0);
                    }
                    else if (i == (PF_DYN2_N_k_grid-1))
                    {
                        (*PF_DYN2_k_D_grid)(i,0)    = (*PF_DYN2_k_grid)(i,0) - (*PF_DYN2_k_grid)(i-1,0);
                    }
                }
            }
            else
            {
                *PF_DYN2_k_grid             = PF_DYN2_k_high * Eigen::RowVectorXd::Ones(PF_DYN2_N_beta_grid);
                (*PF_DYN2_k_D_grid)(0,0)    = DEFAULT_KERNEL_SPREAD;
            }
            /*if (distance_kernel_scale_grid_mode == _QUADRATIC_)
            {
                *PF_DYN2_k_grid     = PF_DYN2_k_grid->array().square();
            }
            else if (distance_kernel_scale_grid_mode == _INVERSE_QUADRATIC_)
            {
                *PF_DYN2_k_grid     = PF_DYN2_k_grid->array().inverse().array().square();
            }*/
            if (PF_DYN2_N_beta_grid > 1)
            {
                PF_DYN2_k_D_grid->block(0,1,PF_DYN2_N_k_grid,PF_DYN2_N_beta_grid-1) = PF_DYN2_k_D_grid->block(0,0,PF_DYN2_N_k_grid,1).replicate(1,
                                                                                                                                                PF_DYN2_N_beta_grid-1);
            }
            (*PF_DYN2_k_D_grid)     = (0.55 * PF_DYN2_k_D_grid->array()).array().square().cwiseInverse();
            *PF_DYN2_k_vector       = Eigen::Map<Eigen::MatrixXd>(PF_DYN2_k_grid->data(),
                                                                  PF_DYN2_N_k_vector,1);
            *PF_DYN2_k_D_vector     = Eigen::Map<Eigen::MatrixXd>(PF_DYN2_k_D_grid->data(),
                                                                  PF_DYN2_N_k_vector,1);
        }
        else // if NOT using PF_DYN2 features
        {
            is_using_PF_DYN2_feature        = false;

            PF_DYN2_feature_matrix_start_idx= AF_H14_feature_matrix_end_idx;
            PF_DYN2_feature_matrix_end_idx  = AF_H14_feature_matrix_end_idx;
            PF_DYN2_feature_matrix_rows     = 0;

            PF_DYN2_N_beta_grid             = 0;
            PF_DYN2_N_k_grid                = 0;
            PF_DYN2_N_beta_vector           = 0;
            PF_DYN2_N_k_vector              = 0;
            PF_DYN2_beta_low                = 0;
            PF_DYN2_beta_high               = 0;
            PF_DYN2_k_low                   = 0;
            PF_DYN2_k_high                  = 0;
        }

        if ((KGF_feature_method > _KGF_NO_USE_) &&
            (KGF_feature_method <= _NUM_KGF_FEATURE_METHODS_))
        {
            is_using_KGF_feature            = true;

            KGF_feature_matrix_start_idx    = PF_DYN2_feature_matrix_end_idx + 1;

            // compute KGF feature parameters: beta, k, and s
            KGF_N_beta_vector               = KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid;
            KGF_N_k_vector                  = KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid;
            KGF_N_s_vector                  = KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid;

            // compute feature size based on selected feature method:
            KGF_feature_matrix_rows         = (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid);
            KGF_feature_matrix_end_idx      = KGF_feature_matrix_start_idx +
                                              KGF_feature_matrix_rows - 1;

            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_col_grid,
                                        KGF_N_beta_grid, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_col_D_grid,
                                        KGF_N_beta_grid, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_row_grid,
                                        KGF_N_k_grid, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_row_D_grid,
                                        KGF_N_k_grid, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_s_depth_grid,
                                        KGF_N_s_grid, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_s_depth_D_grid,
                                        KGF_N_s_grid, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_row_col_grid,
                                        KGF_N_k_grid, KGF_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_row_col_D_grid,
                                        KGF_N_k_grid, KGF_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_rowcol_vector,
                                        (KGF_N_k_grid * KGF_N_beta_grid), 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_rowcol_D_vector,
                                        (KGF_N_k_grid * KGF_N_beta_grid), 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_row_col_grid,
                                        KGF_N_k_grid, KGF_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_row_col_D_grid,
                                        KGF_N_k_grid, KGF_N_beta_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_rowcol_vector,
                                        (KGF_N_k_grid * KGF_N_beta_grid), 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_rowcol_D_vector,
                                        (KGF_N_k_grid * KGF_N_beta_grid), 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_rowcol_depth_grid,
                                        (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_rowcol_depth_D_grid,
                                        (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_rowcoldepth_vector,
                                        KGF_N_beta_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_beta_rowcoldepth_D_vector,
                                        KGF_N_beta_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_rowcol_depth_grid,
                                        (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_rowcol_depth_D_grid,
                                        (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_rowcoldepth_vector,
                                        KGF_N_k_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_k_rowcoldepth_D_vector,
                                        KGF_N_k_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_s_rowcol_depth_grid,
                                        (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_s_rowcol_depth_D_grid,
                                        (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid);
            allocateMemoryIfNonRealTime(is_real_time, KGF_s_rowcoldepth_vector,
                                        KGF_N_s_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_s_rowcoldepth_D_vector,
                                        KGF_N_s_vector, 1);
            allocateMemoryIfNonRealTime(is_real_time, KGF_loa_feature_matrix_per_obs_point,
                                        3, KGF_N_beta_vector);

            if (KGF_N_beta_grid > 1)
            {
                *KGF_beta_col_grid      = Eigen::VectorXd::LinSpaced(KGF_N_beta_grid,
                                                                     KGF_beta_low,
                                                                     KGF_beta_high);
                *KGF_beta_row_col_grid  = Eigen::RowVectorXd::LinSpaced(KGF_N_beta_grid,
                                                                        KGF_beta_low,
                                                                        KGF_beta_high).replicate(KGF_N_k_grid,
                                                                                                 1);
                for (uint j=0; j<KGF_N_beta_grid; j++)
                {
                    if (j < (KGF_N_beta_grid-1))
                    {
                        (*KGF_beta_row_col_D_grid)(0,j) = (*KGF_beta_row_col_grid)(0,j+1) - (*KGF_beta_row_col_grid)(0,j);
                    }
                    else if (j == (KGF_N_beta_grid-1))
                    {
                        (*KGF_beta_row_col_D_grid)(0,j) = (*KGF_beta_row_col_grid)(0,j) - (*KGF_beta_row_col_grid)(0,j-1);
                    }
                }
            }
            else
            {
                (*KGF_beta_col_grid)(0,0)       = KGF_beta_low;
                *KGF_beta_row_col_grid          = KGF_beta_low * Eigen::VectorXd::Ones(KGF_N_k_grid);
                (*KGF_beta_row_col_D_grid)(0,0) = DEFAULT_KERNEL_SPREAD;
            }
            if (KGF_N_k_grid > 1)
            {
                KGF_beta_row_col_D_grid->block(1,0,KGF_N_k_grid-1,KGF_N_beta_grid)  = KGF_beta_row_col_D_grid->block(0,0,1,KGF_N_beta_grid).replicate(KGF_N_k_grid-1,
                                                                                                                                                      1);
            }
            (*KGF_beta_row_col_D_grid)  = (0.55 * KGF_beta_row_col_D_grid->array()).array().square().cwiseInverse();
            *KGF_beta_rowcol_vector     = Eigen::Map<Eigen::MatrixXd>(KGF_beta_row_col_grid->data(),
                                                                      (KGF_N_k_grid * KGF_N_beta_grid),1);
            *KGF_beta_rowcol_D_vector   = Eigen::Map<Eigen::MatrixXd>(KGF_beta_row_col_D_grid->data(),
                                                                      (KGF_N_k_grid * KGF_N_beta_grid),1);
            *KGF_beta_col_D_grid        = KGF_beta_row_col_D_grid->block(0,0,1,KGF_N_beta_grid).transpose();

            if (KGF_N_k_grid > 1)
            {
                *KGF_k_row_grid         = Eigen::VectorXd::LinSpaced(KGF_N_k_grid,
                                                                     KGF_k_low,
                                                                     KGF_k_high);
                *KGF_k_row_col_grid     = Eigen::VectorXd::LinSpaced(KGF_N_k_grid,
                                                                     KGF_k_low,
                                                                     KGF_k_high).replicate(1,
                                                                                           KGF_N_beta_grid);
                if (distance_kernel_scale_grid_mode == _QUADRATIC_)
                {
                    *KGF_k_row_grid             = KGF_k_row_grid->array().square();
                    *KGF_k_row_col_grid         = KGF_k_row_col_grid->array().square();
                }
                else if (distance_kernel_scale_grid_mode == _INVERSE_QUADRATIC_)
                {
                    *KGF_k_row_grid             = KGF_k_row_grid->array().inverse().array().square();
                    *KGF_k_row_col_grid         = KGF_k_row_col_grid->array().inverse().array().square();
                }
                for (uint i=0; i<KGF_N_k_grid; i++)
                {
                    if (i < (KGF_N_k_grid-1))
                    {
                        (*KGF_k_row_col_D_grid)(i,0)    = (*KGF_k_row_col_grid)(i+1,0) - (*KGF_k_row_col_grid)(i,0);
                    }
                    else if (i == (KGF_N_k_grid-1))
                    {
                        (*KGF_k_row_col_D_grid)(i,0)    = (*KGF_k_row_col_grid)(i,0) - (*KGF_k_row_col_grid)(i-1,0);
                    }
                }
            }
            else
            {
                (*KGF_k_row_grid)(0,0)          = KGF_k_high;
                *KGF_k_row_col_grid             = KGF_k_high * Eigen::RowVectorXd::Ones(KGF_N_beta_grid);

                if (distance_kernel_scale_grid_mode == _QUADRATIC_)
                {
                    *KGF_k_row_grid             = KGF_k_row_grid->array().square();
                    *KGF_k_row_col_grid         = KGF_k_row_col_grid->array().square();
                }
                else if (distance_kernel_scale_grid_mode == _INVERSE_QUADRATIC_)
                {
                    *KGF_k_row_grid             = KGF_k_row_grid->array().inverse().array().square();
                    *KGF_k_row_col_grid         = KGF_k_row_col_grid->array().inverse().array().square();
                }

                (*KGF_k_row_col_D_grid)(0,0)    = DEFAULT_KERNEL_SPREAD;
            }
            if (KGF_N_beta_grid > 1)
            {
                KGF_k_row_col_D_grid->block(0,1,KGF_N_k_grid,KGF_N_beta_grid-1) = KGF_k_row_col_D_grid->block(0,0,KGF_N_k_grid,1).replicate(1,
                                                                                                                                            KGF_N_beta_grid-1);
            }
            (*KGF_k_row_col_D_grid)     = (0.55 * KGF_k_row_col_D_grid->array()).array().square().cwiseInverse();
            *KGF_k_rowcol_vector        = Eigen::Map<Eigen::MatrixXd>(KGF_k_row_col_grid->data(),
                                                                      (KGF_N_k_grid * KGF_N_beta_grid),1);
            *KGF_k_rowcol_D_vector      = Eigen::Map<Eigen::MatrixXd>(KGF_k_row_col_D_grid->data(),
                                                                      (KGF_N_k_grid * KGF_N_beta_grid),1);
            *KGF_k_row_D_grid           = KGF_k_row_col_D_grid->block(0,0,KGF_N_k_grid,1);

            *KGF_beta_rowcol_depth_grid     = KGF_beta_rowcol_vector->replicate(1, KGF_N_s_grid);
            *KGF_beta_rowcol_depth_D_grid   = KGF_beta_rowcol_D_vector->replicate(1, KGF_N_s_grid);
            *KGF_beta_rowcoldepth_vector    = Eigen::Map<Eigen::MatrixXd>(KGF_beta_rowcol_depth_grid->data(),
                                                                          KGF_N_beta_vector,1);
            *KGF_beta_rowcoldepth_D_vector  = Eigen::Map<Eigen::MatrixXd>(KGF_beta_rowcol_depth_D_grid->data(),
                                                                          KGF_N_beta_vector,1);

            *KGF_k_rowcol_depth_grid        = KGF_k_rowcol_vector->replicate(1, KGF_N_s_grid);
            *KGF_k_rowcol_depth_D_grid      = KGF_k_rowcol_D_vector->replicate(1, KGF_N_s_grid);
            *KGF_k_rowcoldepth_vector       = Eigen::Map<Eigen::MatrixXd>(KGF_k_rowcol_depth_grid->data(),
                                                                          KGF_N_k_vector,1);
            *KGF_k_rowcoldepth_D_vector     = Eigen::Map<Eigen::MatrixXd>(KGF_k_rowcol_depth_D_grid->data(),
                                                                          KGF_N_k_vector,1);

            if (KGF_N_s_grid > 1)
            {
                /* *KGF_s_rowcol_depth_grid    = Eigen::RowVectorXd::LinSpaced(KGF_N_s_grid,
                                                                            KGF_s_low,
                                                                            KGF_s_high).replicate((KGF_N_k_grid * KGF_N_beta_grid),
                                                                                                  1);*/
                uint    canonical_order = canonical_sys_discrete->getOrder();
                double  alpha_canonical = canonical_sys_discrete->getAlpha();

                // centers are spanned evenly within 0.5 seconds period of
                // the evolution of the canonical state position:
                //double  dt              = ((1.0-0.0)/(model_size-1)) * 0.5;
                double  dt              = ((KGF_s_high-KGF_s_low)/(KGF_N_s_grid-1)) * 0.5;

                // distribute Gaussian centers within 0.5 seconds period of
                // the decaying-exponential evolution of the canonical state position:
                for(uint m = 0; m < KGF_N_s_grid; m++)
                {
                    double  t           = m * dt;
                    if (canonical_order == 2)
                    {
                        // analytical solution to differential equation:
                        // tau^2 * ydd(t) = alpha * ((beta * (0.0 - y(t))) - (tau * yd(t)))
                        // with initial conditions y(0) = 1.0 and yd(0) = 0.0
                        // beta = alpha / 4.0 (for critical damping response on the 2nd order system)
                        // tau = 1.0 and centers[m] = y(m)
                        (*KGF_s_rowcol_depth_grid)(0,m) = (1.0+((alpha_canonical/2.0)*t))*exp(-(alpha_canonical/2.0)*t);
                    }
                    else	// 1st order canonical system
                    {
                        // analytical solution to differential equation:
                        // tau * yd(t) = -alpha * y(t)
                        // with initial conditions y(0) = 1.0
                        // tau = 1.0 and centers[m] = y(m)
                        (*KGF_s_rowcol_depth_grid)(0,m) = exp(-alpha_canonical * t);
                    }
                }
                *KGF_s_depth_grid   = KGF_s_rowcol_depth_grid->block(0,0,1,KGF_N_s_grid).transpose();
                KGF_s_rowcol_depth_grid->block(1,0,(KGF_N_k_grid * KGF_N_beta_grid)-1,KGF_N_s_grid)   = KGF_s_rowcol_depth_grid->block(0,0,1,KGF_N_s_grid).replicate((KGF_N_k_grid * KGF_N_beta_grid)-1,
                                                                                                                                                                     1);

                for (uint j=0; j<KGF_N_s_grid; j++)
                {
                    if (j < (KGF_N_s_grid-1))
                    {
                        (*KGF_s_rowcol_depth_D_grid)(0,j)   = (*KGF_s_rowcol_depth_grid)(0,j+1) - (*KGF_s_rowcol_depth_grid)(0,j);
                    }
                    else if (j == (KGF_N_s_grid-1))
                    {
                        (*KGF_s_rowcol_depth_D_grid)(0,j)   = (*KGF_s_rowcol_depth_grid)(0,j) - (*KGF_s_rowcol_depth_grid)(0,j-1);
                    }
                }
            }
            else
            {
                (*KGF_s_depth_grid)(0,0)            = KGF_s_low;
                *KGF_s_rowcol_depth_grid            = KGF_s_low * Eigen::VectorXd::Ones(KGF_N_k_grid * KGF_N_beta_grid);
                (*KGF_s_rowcol_depth_D_grid)(0,0)   = DEFAULT_KERNEL_SPREAD;
            }
            if ((KGF_N_k_grid * KGF_N_beta_grid) > 1)
            {
                KGF_s_rowcol_depth_D_grid->block(1,0,(KGF_N_k_grid * KGF_N_beta_grid)-1,KGF_N_s_grid)   = KGF_s_rowcol_depth_D_grid->block(0,0,1,KGF_N_s_grid).replicate((KGF_N_k_grid * KGF_N_beta_grid)-1,
                                                                                                                                                                         1);
            }
            (*KGF_s_rowcol_depth_D_grid)    = (0.55 * KGF_s_rowcol_depth_D_grid->array()).array().square().cwiseInverse();
            *KGF_s_rowcoldepth_vector       = Eigen::Map<Eigen::MatrixXd>(KGF_s_rowcol_depth_grid->data(),
                                                                          KGF_N_s_vector,1);
            *KGF_s_rowcoldepth_D_vector     = Eigen::Map<Eigen::MatrixXd>(KGF_s_rowcol_depth_D_grid->data(),
                                                                          KGF_N_s_vector,1);
            *KGF_s_depth_D_grid             = KGF_s_rowcol_depth_D_grid->block(0,0,1,KGF_N_s_grid).transpose();
        }
        else // if NOT using KGF features
        {
            is_using_KGF_feature        = false;

            KGF_feature_matrix_start_idx= PF_DYN2_feature_matrix_end_idx;
            KGF_feature_matrix_end_idx  = PF_DYN2_feature_matrix_end_idx;
            KGF_feature_matrix_rows     = 0;

            KGF_N_beta_grid             = 0;
            KGF_N_k_grid                = 0;
            KGF_N_s_grid                = 0;
            KGF_N_beta_vector           = 0;
            KGF_N_k_vector              = 0;
            KGF_N_s_vector              = 0;
            KGF_beta_low                = 0;
            KGF_beta_high               = 0;
            KGF_k_low                   = 0;
            KGF_k_high                  = 0;
            KGF_s_low                   = 0;
            KGF_s_high                  = 0;
        }

        feature_vector_size         = 3 * (AF_H14_feature_matrix_rows +
                                           PF_DYN2_feature_matrix_rows +
                                           KGF_feature_matrix_rows);

        allocateMemoryIfNonRealTime(is_real_time, parameter_beta_matrix,
                                    (feature_vector_size/3), 3);
        allocateMemoryIfNonRealTime(is_real_time, parameter_k_matrix,
                                    (feature_vector_size/3), 3);
        allocateMemoryIfNonRealTime(is_real_time, parameter_s_matrix,
                                    (feature_vector_size/3), 3);
        if (is_using_KGF_feature)
        {
            allocateMemoryIfNonRealTime(is_real_time, parameter_table,
                                        feature_vector_size, 3);
        }
        else
        {
            allocateMemoryIfNonRealTime(is_real_time, parameter_table,
                                        feature_vector_size, 2);
        }

        allocateMemoryIfNonRealTime(is_real_time, weights,
                                    3, feature_vector_size);
        allocateMemoryIfNonRealTime(is_real_time, weights_Rp_yd_constrained,
                                    (feature_vector_size/3), 1);
        allocateMemoryIfNonRealTime(is_real_time, cart_coord_loa_feature_matrix,
                                    (feature_vector_size/3), 3);

        // Log feature parameters into parameter table:
        (*parameter_beta_matrix)    = ZeroMatrixloaF1x3(feature_vector_size/3);
        (*parameter_k_matrix)       = ZeroMatrixloaF1x3(feature_vector_size/3);
        (*parameter_s_matrix)       = ZeroMatrixloaF1x3(feature_vector_size/3);

        if (is_using_AF_H14_feature)
        {
            for (uint i=0; i<AF_H14_num_indep_feat_obs_points; i++)
            {
                parameter_beta_matrix->block((AF_H14_feature_matrix_start_idx+(i*AF_H14_beta_phi1_phi2_vector->rows())),0,AF_H14_beta_phi1_phi2_vector->rows(),3)   = AF_H14_beta_phi1_phi2_vector->replicate(1,3);
                parameter_k_matrix->block((AF_H14_feature_matrix_start_idx+(i*AF_H14_k_phi1_phi2_vector->rows())),0,AF_H14_k_phi1_phi2_vector->rows(),3)            = AF_H14_k_phi1_phi2_vector->replicate(1,3);
            }

            if (AF_H14_N_k_phi3_grid > 0)
            {
                parameter_k_matrix->block((AF_H14_feature_matrix_start_idx+(AF_H14_num_indep_feat_obs_points*AF_H14_k_phi1_phi2_vector->rows())),0,AF_H14_k_phi3_vector->rows(),3)  = AF_H14_k_phi3_vector->replicate(1,3);
            }
        }

        if (is_using_PF_DYN2_feature)
        {
            parameter_beta_matrix->block(PF_DYN2_feature_matrix_start_idx,0,PF_DYN2_beta_vector->rows(),3)  = PF_DYN2_beta_vector->replicate(1,3);
            parameter_k_matrix->block(PF_DYN2_feature_matrix_start_idx,0,PF_DYN2_k_vector->rows(),3)        = PF_DYN2_k_vector->replicate(1,3);
        }

        if (is_using_KGF_feature)
        {
            parameter_beta_matrix->block(KGF_feature_matrix_start_idx,0,KGF_beta_rowcoldepth_vector->rows(),3)  = KGF_beta_rowcoldepth_vector->replicate(1,3);
            parameter_k_matrix->block(KGF_feature_matrix_start_idx,0,KGF_k_rowcoldepth_vector->rows(),3)        = KGF_k_rowcoldepth_vector->replicate(1,3);
            parameter_s_matrix->block(KGF_feature_matrix_start_idx,0,KGF_s_rowcoldepth_vector->rows(),3)        = KGF_s_rowcoldepth_vector->replicate(1,3);
        }

        parameter_table->block(0,0,feature_vector_size,1)       = Eigen::Map<MatrixXxX>(parameter_beta_matrix->data(), feature_vector_size, 1);
        parameter_table->block(0,1,feature_vector_size,1)       = Eigen::Map<MatrixXxX>(parameter_k_matrix->data(), feature_vector_size, 1);
        if (is_using_KGF_feature)
        {
            parameter_table->block(0,2,feature_vector_size,1)   = Eigen::Map<MatrixXxX>(parameter_s_matrix->data(), feature_vector_size, 1);
        }
    }
    else if ((NN_feature_method > _NN_NO_USE_) &&
             (NN_feature_method <= _NUM_NN_FEATURE_METHODS_))
    {
        is_using_NN             = true;
        is_using_AF_H14_feature = false;
        is_using_PF_DYN2_feature= false;
        is_using_KGF_feature    = false;

        NN_N_k_grid             = 3;
        NN_k_low                = 0.01;
        NN_k_high               = 0.5;

        allocateMemoryIfNonRealTime(is_real_time, NN_k_grid,
                                    NN_N_k_grid, 1);
        if (NN_N_k_grid > 1)
        {
            *NN_k_grid          = Eigen::VectorXd::LinSpaced(NN_N_k_grid,
                                                             NN_k_low,
                                                             NN_k_high);
        }
        else if (NN_N_k_grid == 1)
        {
            (*NN_k_grid)(0,0)   = NN_k_high;
        }
        if (distance_kernel_scale_grid_mode == _INVERSE_QUADRATIC_)
        {
            *NN_k_grid          = NN_k_grid->array().inverse().array().square();
        }

        allocateMemoryIfNonRealTime(is_real_time, NN_dist_kernels,
                                    NN_N_k_grid, 1);

        NN_feature_vector_rows  = NN_N_input;

        feature_vector_size     = NN_N_input;

        allocateMemoryIfNonRealTime(is_real_time, NN_loa_feature_vector_buffer,
                                    NN_feature_vector_rows/NN_N_k_grid, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_feature_vector,
                                    NN_feature_vector_rows, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_I_x1_step1_xoffset_vector,
                                    NN_feature_vector_rows, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_I_x1_step1_gain_vector,
                                    NN_feature_vector_rows, 1);
        NN_loa_I_x1_step1_ymin  = 0.0;
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_I_Xp1_vector,
                                    NN_feature_vector_rows, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_H1xI__IW1_1_weight_matrix,
                                    NN_N_hidden_layer_1, NN_N_input);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_H1_b1_bias_vector,
                                    NN_N_hidden_layer_1, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_H1_a1_vector,
                                    NN_N_hidden_layer_1, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_H2xH1_LW2_1_weight_matrix,
                                    NN_N_hidden_layer_2, NN_N_hidden_layer_1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_H2_b2_bias_vector,
                                    NN_N_hidden_layer_2, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_H2_a2_vector,
                                    NN_N_hidden_layer_2, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_OxH2__LW3_2_weight_matrix,
                                    NN_N_output, NN_N_hidden_layer_2);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_O_b3_bias_vector,
                                    NN_N_output, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_O_a3_vector,
                                    NN_N_output, 1);
        NN_loa_O_y1_step1_ymin  = 0.0;
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_O_y1_step1_gain_vector,
                                    NN_N_output, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_O_y1_step1_xoffset_vector,
                                    NN_N_output, 1);
        allocateMemoryIfNonRealTime(is_real_time, NN_loa_O_y_vector,
                                    NN_N_output, 1);

        loa_data_io.readMatrixFromFile(NN_params_directory_path, "b1.txt", *NN_loa_H1_b1_bias_vector);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "b2.txt", *NN_loa_H2_b2_bias_vector);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "b3.txt", *NN_loa_O_b3_bias_vector);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "IW1_1.txt", *NN_loa_H1xI__IW1_1_weight_matrix);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "LW2_1.txt", *NN_loa_H2xH1_LW2_1_weight_matrix);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "LW3_2.txt", *NN_loa_OxH2__LW3_2_weight_matrix);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "x1_step1_gain.txt", *NN_loa_I_x1_step1_gain_vector);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "x1_step1_xoffset.txt", *NN_loa_I_x1_step1_xoffset_vector);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "x1_step1_ymin.txt", NN_loa_I_x1_step1_ymin);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "y1_step1_gain.txt", *NN_loa_O_y1_step1_gain_vector);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "y1_step1_xoffset.txt", *NN_loa_O_y1_step1_xoffset_vector);
        loa_data_io.readMatrixFromFile(NN_params_directory_path, "y1_step1_ymin.txt", NN_loa_O_y1_step1_ymin);
    }

    allocateMemoryIfNonRealTime(is_real_time, cart_coord_loa_feature_vector,
                                feature_vector_size, 1);
}

/**
 * Checks the validity of this data structure.
 *
 * @return Valid (true) or invalid (false)
 */
bool TCLearnObsAvoidFeatureParameter::isValid()
{
    if (rt_assert((rt_assert(feature_vector_size >  0)) &&
                  (rt_assert(feature_vector_size <= MAX_LOA_NUM_FEATURE_VECTOR_SIZE))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(data_file_format >  0)) &&
                  (rt_assert(data_file_format <= _NUM_DATA_FILE_FORMATS_))) == false)
    {
        return false;
    }
    if (rt_assert((rt_assert(distance_kernel_scale_grid_mode >  0)) &&
                  (rt_assert(distance_kernel_scale_grid_mode <= _NUM_DISTANCE_KERNEL_SCALE_GRID_MODES_))) == false)
    {
        return false;
    }
    if (rt_assert(isMemoryAllocated(cart_coord_loa_feature_vector,
                                    feature_vector_size, 1)) == false)
    {
        return false;
    }
    if (is_using_NN == false)   // not using neural network
    {
        if (rt_assert(isMemoryAllocated(weights,
                                        3, feature_vector_size)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(weights_Rp_yd_constrained,
                                        (feature_vector_size/3), 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(cart_coord_loa_feature_matrix,
                                        (feature_vector_size/3), 3)) == false)
        {
            return false;
        }
        if (is_using_AF_H14_feature)
        {
            if (rt_assert((rt_assert(AF_H14_feature_method >  0)) &&
                          (rt_assert(AF_H14_feature_method <= _NUM_AF_H14_FEATURE_METHODS_))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(AF_H14_feature_matrix_rows >  0)) &&
                          (rt_assert(AF_H14_feature_matrix_rows <= MAX_AF_H14_NUM_FEATURE_MATRIX_ROWS))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(AF_H14_num_indep_feat_obs_points >  0)) &&
                          (rt_assert(AF_H14_num_indep_feat_obs_points <= MAX_AF_H14_NUM_INDEP_FEAT_OBS_POINTS))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(AF_H14_N_beta_phi1_phi2_grid >  0)) &&
                          (rt_assert(AF_H14_N_beta_phi1_phi2_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(AF_H14_N_k_phi1_phi2_grid >  0)) &&
                          (rt_assert(AF_H14_N_k_phi1_phi2_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(AF_H14_N_k_phi3_grid >= 0)) &&
                          (rt_assert(AF_H14_N_k_phi3_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((AF_H14_N_beta_phi1_phi2_vector == (AF_H14_N_k_phi1_phi2_grid * AF_H14_N_beta_phi1_phi2_grid)) &&
                          (AF_H14_N_k_phi1_phi2_vector    == (AF_H14_N_k_phi1_phi2_grid * AF_H14_N_beta_phi1_phi2_grid))) == false)
            {
                return false;
            }
            if (rt_assert((AF_H14_beta_phi1_phi2_low    >= 0) &&
                          (AF_H14_beta_phi1_phi2_high   >= AF_H14_beta_phi1_phi2_low) &&
                          (AF_H14_k_phi1_phi2_low       >= 0) &&
                          (AF_H14_k_phi1_phi2_high      >= AF_H14_k_phi1_phi2_low) &&
                          (AF_H14_k_phi3_low            >= 0) &&
                          (AF_H14_k_phi3_high           >= AF_H14_k_phi3_low)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(AF_H14_beta_phi1_phi2_grid,
                                            AF_H14_N_k_phi1_phi2_grid,
                                            AF_H14_N_beta_phi1_phi2_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(AF_H14_k_phi1_phi2_grid,
                                            AF_H14_N_k_phi1_phi2_grid,
                                            AF_H14_N_beta_phi1_phi2_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(AF_H14_beta_phi1_phi2_vector,
                                            (AF_H14_N_k_phi1_phi2_grid * AF_H14_N_beta_phi1_phi2_grid),
                                            1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(AF_H14_k_phi1_phi2_vector,
                                            (AF_H14_N_k_phi1_phi2_grid * AF_H14_N_beta_phi1_phi2_grid),
                                            1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point,
                                            3, AF_H14_N_beta_phi1_phi2_vector)) == false)
            {
                return false;
            }
            if (AF_H14_N_k_phi3_grid > 0)
            {
                if (rt_assert(isMemoryAllocated(AF_H14_k_phi3_vector,
                                                AF_H14_N_k_phi3_grid, 1)) == false)
                {
                    return false;
                }
                if (rt_assert(isMemoryAllocated(AF_H14_loa_feature_matrix_phi3_per_obs_point,
                                                3, AF_H14_N_k_phi3_grid)) == false)
                {
                    return false;
                }
            }
            if (rt_assert(AF_H14_beta_phi1_phi2_vector->rows() ==
                          AF_H14_k_phi1_phi2_vector->rows()) == false)
            {
                return false;
            }
        }
        if (is_using_PF_DYN2_feature)
        {
            if (rt_assert((rt_assert(PF_DYN2_feature_method >  0)) &&
                          (rt_assert(PF_DYN2_feature_method <= _NUM_PF_DYN2_FEATURE_METHODS_))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(PF_DYN2_feature_matrix_rows >  0)) &&
                          (rt_assert(PF_DYN2_feature_matrix_rows <= MAX_PF_DYN2_NUM_FEATURE_MATRIX_ROWS))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(PF_DYN2_N_beta_grid >  0)) &&
                          (rt_assert(PF_DYN2_N_beta_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(PF_DYN2_N_k_grid >  0)) &&
                          (rt_assert(PF_DYN2_N_k_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((PF_DYN2_feature_matrix_rows == (PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid)) &&
                          (PF_DYN2_N_beta_vector       == (PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid)) &&
                          (PF_DYN2_N_k_vector          == (PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid))) == false)
            {
                return false;
            }
            if (rt_assert((PF_DYN2_beta_high    >= PF_DYN2_beta_low) &&
                          (PF_DYN2_k_high       >= PF_DYN2_k_low)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_beta_grid,
                                            PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_beta_D_grid,
                                            PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_k_grid,
                                            PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_k_D_grid,
                                            PF_DYN2_N_k_grid, PF_DYN2_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_beta_vector,
                                            (PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_beta_D_vector,
                                            (PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_k_vector,
                                            (PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_k_D_vector,
                                            (PF_DYN2_N_k_grid * PF_DYN2_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(PF_DYN2_loa_feature_matrix_per_obs_point,
                                            3, PF_DYN2_N_beta_vector)) == false)
            {
                return false;
            }
            if (rt_assert(PF_DYN2_beta_vector->rows() == PF_DYN2_k_vector->rows()) == false)
            {
                return false;
            }
        }
        if (is_using_KGF_feature)
        {
            if (rt_assert((rt_assert(KGF_feature_method >  0)) &&
                          (rt_assert(KGF_feature_method <= _NUM_KGF_FEATURE_METHODS_))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(KGF_feature_matrix_rows >  0)) &&
                          (rt_assert(KGF_feature_matrix_rows <= MAX_KGF_NUM_FEATURE_MATRIX_ROWS))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(KGF_N_beta_grid >  0)) &&
                          (rt_assert(KGF_N_beta_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(KGF_N_k_grid >  0)) &&
                          (rt_assert(KGF_N_k_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(KGF_N_s_grid >  0)) &&
                          (rt_assert(KGF_N_s_grid <= MAX_LOA_GRID_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((KGF_feature_matrix_rows == (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid)) &&
                          (KGF_N_beta_vector       == (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid)) &&
                          (KGF_N_k_vector          == (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid)) &&
                          (KGF_N_s_vector          == (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid))) == false)
            {
                return false;
            }
            if (rt_assert((KGF_beta_high >= KGF_beta_low) &&
                          (KGF_k_high    >= KGF_k_low)    &&
                          (KGF_s_high    >= KGF_s_low)) == false)
            {
                return false;
            }

            if (rt_assert(isMemoryAllocated(KGF_beta_col_grid,
                                            KGF_N_beta_grid, 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_col_D_grid,
                                            KGF_N_beta_grid, 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_row_grid,
                                            KGF_N_k_grid, 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_row_D_grid,
                                            KGF_N_k_grid, 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_s_depth_grid,
                                            KGF_N_s_grid, 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_s_depth_D_grid,
                                            KGF_N_s_grid, 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_row_col_grid,
                                            KGF_N_k_grid, KGF_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_row_col_D_grid,
                                            KGF_N_k_grid, KGF_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_rowcol_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_rowcol_D_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_row_col_grid,
                                            KGF_N_k_grid, KGF_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_row_col_D_grid,
                                            KGF_N_k_grid, KGF_N_beta_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_rowcol_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_rowcol_D_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_rowcol_depth_grid,
                                            (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_rowcol_depth_D_grid,
                                            (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_rowcoldepth_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_beta_rowcoldepth_D_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_rowcol_depth_grid,
                                            (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_rowcol_depth_D_grid,
                                            (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_rowcoldepth_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_k_rowcoldepth_D_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_s_rowcol_depth_grid,
                                            (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_s_rowcol_depth_D_grid,
                                            (KGF_N_k_grid * KGF_N_beta_grid), KGF_N_s_grid)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_s_rowcoldepth_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_s_rowcoldepth_D_vector,
                                            (KGF_N_k_grid * KGF_N_beta_grid * KGF_N_s_grid), 1)) == false)
            {
                return false;
            }
            if (rt_assert(isMemoryAllocated(KGF_loa_feature_matrix_per_obs_point,
                                            3, KGF_N_beta_vector)) == false)
            {
                return false;
            }
            if (rt_assert((KGF_beta_rowcoldepth_vector->rows() == KGF_k_rowcoldepth_vector->rows()) &&
                          (KGF_k_rowcoldepth_vector->rows()    == KGF_s_rowcoldepth_vector->rows())) == false)
            {
                return false;
            }
        }
    }
    else    // if (is_using_NN == true) (using neural network)
    {
        if (rt_assert((rt_assert(NN_N_input >  0)) &&
                      (rt_assert(NN_N_input <= MAX_NN_NUM_INPUT_NODES))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(NN_N_hidden_layer_1 >  0)) &&
                      (rt_assert(NN_N_hidden_layer_1 <= MAX_NN_NUM_HIDDEN_NODES))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(NN_N_hidden_layer_2 >  0)) &&
                      (rt_assert(NN_N_hidden_layer_2 <= MAX_NN_NUM_HIDDEN_NODES))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(NN_N_output >  0)) &&
                      (rt_assert(NN_N_output <= MAX_NN_NUM_OUTPUT_NODES))) == false)
        {
            return false;
        }
        if (rt_assert((rt_assert(NN_feature_method >  0)) &&
                      (rt_assert(NN_feature_method <= _NUM_NN_FEATURE_METHODS_))) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_k_grid,
                                        NN_N_k_grid, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_dist_kernels,
                                        NN_N_k_grid, 1)) == false)
        {
            return false;
        }
        if (rt_assert(NN_feature_vector_rows == NN_N_input) == false)
        {
            return false;
        }
        if (rt_assert(feature_vector_size == NN_N_input) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_feature_vector_buffer,
                                        NN_feature_vector_rows/NN_N_k_grid, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_feature_vector,
                                        NN_feature_vector_rows, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_I_x1_step1_xoffset_vector,
                                        NN_feature_vector_rows, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_I_x1_step1_gain_vector,
                                        NN_feature_vector_rows, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_I_Xp1_vector,
                                        NN_feature_vector_rows, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_H1xI__IW1_1_weight_matrix,
                                        NN_N_hidden_layer_1, NN_N_input)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_H1_b1_bias_vector,
                                        NN_N_hidden_layer_1, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_H1_a1_vector,
                                        NN_N_hidden_layer_1, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_H2xH1_LW2_1_weight_matrix,
                                        NN_N_hidden_layer_2, NN_N_hidden_layer_1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_H2_b2_bias_vector,
                                        NN_N_hidden_layer_2, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_H2_a2_vector,
                                        NN_N_hidden_layer_2, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_OxH2__LW3_2_weight_matrix,
                                        NN_N_output, NN_N_hidden_layer_2)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_O_b3_bias_vector,
                                        NN_N_output, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_O_a3_vector,
                                        NN_N_output, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_O_y1_step1_gain_vector,
                                        NN_N_output, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_O_y1_step1_xoffset_vector,
                                        NN_N_output, 1)) == false)
        {
            return false;
        }
        if (rt_assert(isMemoryAllocated(NN_loa_O_y_vector,
                                        NN_N_output, 1)) == false)
        {
            return false;
        }
        if (pmnn != NULL)
        {
            if (rt_assert(pmnn->isValid()) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(pmnn_input_vector != NULL)) &&
                          (rt_assert(pmnn_phase_kernel_modulation != NULL)) &&
                          (rt_assert(pmnn_output_vector != NULL))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(pmnn_input_vector->rows() >= 0)) &&
                          (rt_assert(pmnn_input_vector->rows() <= MAX_NN_NUM_NODES_PER_LAYER))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(pmnn_phase_kernel_modulation->rows() >= 0)) &&
                          (rt_assert(pmnn_phase_kernel_modulation->rows() <= MAX_MODEL_SIZE))) == false)
            {
                return false;
            }
            if (rt_assert((rt_assert(pmnn_output_vector->rows() >= 0)) &&
                          (rt_assert(pmnn_output_vector->rows() <= 3))) == false)
            {
                return false;
            }
        }
    }
    return true;
}

TCLearnObsAvoidFeatureParameter::~TCLearnObsAvoidFeatureParameter()
{}

}
