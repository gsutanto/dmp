/*
 * TCLearnObsAvoidFeatureParameter.h
 *
 *  Class defining feature parameters data structure for transform coupling term of 
 *  obstacle avoidance from learning by demonstration on a DMP.
 *
 *  See A. Rai, F. Meier, A. Ijspeert, and S. Schaal;
 *      Learning coupling terms for obstacle avoidance;
 *      IEEE-RAS International Conference on Humanoid Robotics, pp. 512-518, November 2014
 *
 *  Created on: Jul 22, 2016
 *  Author: Giovanni Sutanto
 */

#ifndef TC_LEARN_OBS_AVOID_FEATURE_PARAMETER_H
#define TC_LEARN_OBS_AVOID_FEATURE_PARAMETER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <Eigen/Dense>

#include "dmp/utility/utility.h"
#include "dmp/dmp_state/DMPState.h"
#include "dmp/dmp_discrete/CanonicalSystemDiscrete.h"
#include "dmp/dmp_coupling/learn_obs_avoid/DefinitionsLearnObsAvoid.h"
#include "dmp/dmp_coupling/learn_obs_avoid/LearnObsAvoidDataIO.h"
#include "dmp/neural_nets/PMNN.h"

namespace dmp
{

class TCLearnObsAvoidFeatureParameter
{

    friend class                TransformCouplingLearnObsAvoid;

protected:

    LearnObsAvoidDataIO         loa_data_io;

    RealTimeAssertor*           rt_assertor;
    CanonicalSystemDiscrete*    canonical_sys_discrete;

private:

    uint                        feature_vector_size;
    uint                        data_file_format;
    uint                        distance_kernel_scale_grid_mode;

    // The following field(s) are written as SMART pointers to reduce size overhead
    // in the stack, by allocating them in the heap:
    MatrixloaF1x3Ptr            parameter_beta_matrix;
    MatrixloaF1x3Ptr            parameter_k_matrix;
    MatrixloaF1x3Ptr            parameter_s_matrix;
    MatrixloaF3x3Ptr            parameter_table;

    Matrixloa3xF3Ptr            weights;
    VectorloaF1Ptr              weights_Rp_yd_constrained;
    MatrixloaF1x3Ptr            cart_coord_loa_feature_matrix;
    VectorloaF3Ptr              cart_coord_loa_feature_vector;



    // Akshara-Franzi Humanoids'14 (AF_H14) Model Parameters:
    bool                        is_using_AF_H14_feature;

    uint                        AF_H14_feature_method;
    int                         AF_H14_feature_matrix_start_idx;
    int                         AF_H14_feature_matrix_end_idx;
    uint                        AF_H14_feature_matrix_rows;
    uint                        AF_H14_num_indep_feat_obs_points;   // number of obstacle points that will be considered/evaluated as separate/independent feature sets

    uint                        AF_H14_N_beta_phi1_phi2_grid;
    uint                        AF_H14_N_k_phi1_phi2_grid;
    uint                        AF_H14_N_k_phi3_grid;

    uint                        AF_H14_N_beta_phi1_phi2_vector;
    uint                        AF_H14_N_k_phi1_phi2_vector;

    double                      AF_H14_beta_phi1_phi2_low;
    double                      AF_H14_beta_phi1_phi2_high;
    double                      AF_H14_k_phi1_phi2_low;
    double                      AF_H14_k_phi1_phi2_high;
    double                      AF_H14_k_phi3_low;
    double                      AF_H14_k_phi3_high;

    // The following field(s) are written as SMART pointers to reduce size overhead
    // in the stack, by allocating them in the heap:
    MatrixloaGxGPtr             AF_H14_beta_phi1_phi2_grid;
    MatrixloaGxGPtr             AF_H14_k_phi1_phi2_grid;
    VectorloaG2Ptr              AF_H14_beta_phi1_phi2_vector;
    VectorloaG2Ptr              AF_H14_k_phi1_phi2_vector;
    VectorloaGPtr               AF_H14_k_phi3_vector;
    Matrixloa3xG2Ptr            AF_H14_loa_feature_matrix_phi1_phi2_per_obs_point;
    Matrixloa3xGPtr             AF_H14_loa_feature_matrix_phi3_per_obs_point;



    // Potential Function Dynamic (PF_DYN2) Model Parameters:
    bool                        is_using_PF_DYN2_feature;

    uint                        PF_DYN2_feature_method;
    int                         PF_DYN2_feature_matrix_start_idx;
    int                         PF_DYN2_feature_matrix_end_idx;
    uint                        PF_DYN2_feature_matrix_rows;

    uint                        PF_DYN2_N_beta_grid;
    uint                        PF_DYN2_N_k_grid;

    uint                        PF_DYN2_N_beta_vector;
    uint                        PF_DYN2_N_k_vector;

    double                      PF_DYN2_beta_low;
    double                      PF_DYN2_beta_high;
    double                      PF_DYN2_k_low;
    double                      PF_DYN2_k_high;

    // The following field(s) are written as SMART pointers to reduce size overhead
    // in the stack, by allocating them in the heap:
    MatrixloaGxGPtr             PF_DYN2_beta_grid;      // center of each (cos(theta))-based kernel
    MatrixloaGxGPtr             PF_DYN2_beta_D_grid;    // spread of each (cos(theta))-based kernel
    MatrixloaGxGPtr             PF_DYN2_k_grid;         // center of each distance-from-obstacle-based kernel
    MatrixloaGxGPtr             PF_DYN2_k_D_grid;       // spread of each distance-from-obstacle-based kernel
    VectorloaG2Ptr              PF_DYN2_beta_vector;    // vectorized version of PF_DYN2_beta_grid
    VectorloaG2Ptr              PF_DYN2_beta_D_vector;  // vectorized version of PF_DYN2_beta_D_grid
    VectorloaG2Ptr              PF_DYN2_k_vector;       // vectorized version of PF_DYN2_k_grid
    VectorloaG2Ptr              PF_DYN2_k_D_vector;     // vectorized version of PF_DYN2_k_D_grid
    Matrixloa3xG2Ptr            PF_DYN2_loa_feature_matrix_per_obs_point;



    // Kernelized General Features (KGF) Model Parameters:
    bool                        is_using_KGF_feature;

    uint                        KGF_feature_method;
    int                         KGF_feature_matrix_start_idx;
    int                         KGF_feature_matrix_end_idx;
    uint                        KGF_feature_matrix_rows;

    uint                        KGF_N_beta_grid;    // col
    uint                        KGF_N_k_grid;       // row
    uint                        KGF_N_s_grid;       // depth

    uint                        KGF_N_beta_vector;
    uint                        KGF_N_k_vector;
    uint                        KGF_N_s_vector;

    double                      KGF_beta_low;
    double                      KGF_beta_high;
    double                      KGF_k_low;
    double                      KGF_k_high;
    double                      KGF_s_low;
    double                      KGF_s_high;

    // The following field(s) are written as SMART pointers to reduce size overhead
    // in the stack, by allocating them in the heap:
    VectorloaGPtr               KGF_beta_col_grid;
    VectorloaGPtr               KGF_beta_col_D_grid;
    VectorloaGPtr               KGF_k_row_grid;
    VectorloaGPtr               KGF_k_row_D_grid;
    VectorloaGPtr               KGF_s_depth_grid;
    VectorloaGPtr               KGF_s_depth_D_grid;
    MatrixloaGxGPtr             KGF_beta_row_col_grid;      // center of each (theta)-based kernel
    MatrixloaGxGPtr             KGF_beta_row_col_D_grid;    // spread of each (theta)-based kernel
    VectorloaG2Ptr              KGF_beta_rowcol_vector;     // vectorized version of KGF_beta_grid
    VectorloaG2Ptr              KGF_beta_rowcol_D_vector;   // vectorized version of KGF_beta_D_grid
    MatrixloaGxGPtr             KGF_k_row_col_grid;         // center of each distance-from-obstacle-based kernel
    MatrixloaGxGPtr             KGF_k_row_col_D_grid;       // spread of each distance-from-obstacle-based kernel
    VectorloaG2Ptr              KGF_k_rowcol_vector;        // vectorized version of KGF_k_grid
    VectorloaG2Ptr              KGF_k_rowcol_D_vector;      // vectorized version of KGF_k_D_grid
    MatrixloaG2xGPtr            KGF_beta_rowcol_depth_grid;
    MatrixloaG2xGPtr            KGF_beta_rowcol_depth_D_grid;
    VectorloaG3Ptr              KGF_beta_rowcoldepth_vector;
    VectorloaG3Ptr              KGF_beta_rowcoldepth_D_vector;
    MatrixloaG2xGPtr            KGF_k_rowcol_depth_grid;
    MatrixloaG2xGPtr            KGF_k_rowcol_depth_D_grid;
    VectorloaG3Ptr              KGF_k_rowcoldepth_vector;
    VectorloaG3Ptr              KGF_k_rowcoldepth_D_vector;
    MatrixloaG2xGPtr            KGF_s_rowcol_depth_grid;    // center of each phase-based kernel
    MatrixloaG2xGPtr            KGF_s_rowcol_depth_D_grid;  // spread of each phase-based kernel
    VectorloaG3Ptr              KGF_s_rowcoldepth_vector;   // vectorized version of KGF_s_grid
    VectorloaG3Ptr              KGF_s_rowcoldepth_D_vector; // vectorized version of KGF_s_D_grid
    Matrixloa3xG3Ptr            KGF_loa_feature_matrix_per_obs_point;



    // Neural Network (NN) Model Parameters:
    bool                        is_using_NN;

    uint                        NN_feature_method;
    uint                        NN_feature_vector_rows;

    uint                        NN_N_input;
    uint                        NN_N_hidden_layer_1;
    uint                        NN_N_hidden_layer_2;
    uint                        NN_N_output;

    uint                        NN_N_k_grid;

    double                      NN_k_low;
    double                      NN_k_high;

    VectorloaGPtr               NN_k_grid;
    VectorloaGPtr               NN_dist_kernels;

    VectorloaNN_IPtr            NN_loa_feature_vector_buffer;
    VectorloaNN_IPtr            NN_loa_feature_vector;
    VectorloaNN_IPtr            NN_loa_I_x1_step1_xoffset_vector;
    VectorloaNN_IPtr            NN_loa_I_x1_step1_gain_vector;
    double                      NN_loa_I_x1_step1_ymin;
    VectorloaNN_IPtr            NN_loa_I_Xp1_vector;
    MatrixloaNN_HxIPtr          NN_loa_H1xI__IW1_1_weight_matrix;
    VectorloaNN_HPtr            NN_loa_H1_b1_bias_vector;
    VectorloaNN_HPtr            NN_loa_H1_a1_vector;
    MatrixloaNN_HxHPtr          NN_loa_H2xH1_LW2_1_weight_matrix;
    VectorloaNN_HPtr            NN_loa_H2_b2_bias_vector;
    VectorloaNN_HPtr            NN_loa_H2_a2_vector;
    MatrixloaNN_OxHPtr          NN_loa_OxH2__LW3_2_weight_matrix;
    VectorloaNN_OPtr            NN_loa_O_b3_bias_vector;
    VectorloaNN_OPtr            NN_loa_O_a3_vector;
    double                      NN_loa_O_y1_step1_ymin;
    VectorloaNN_OPtr            NN_loa_O_y1_step1_gain_vector;
    VectorloaNN_OPtr            NN_loa_O_y1_step1_xoffset_vector;
    VectorloaNN_OPtr            NN_loa_O_y_vector;


public:

    PMNN*       pmnn;                           // (pointer to) the PMNN prediction model; this pointer might be shared by other TransformCouplingLearnTactileFeedback object

    VectorNN_N* pmnn_input_vector;              // (pointer to) the input vector; this pointer might be shared by other TransformCouplingLearnTactileFeedback object
    VectorNN_N* pmnn_phase_kernel_modulation;   // (pointer to) the phase kernel modulator, modulating the final hidden layer of the pmnn model; this pointer might be shared by other TransformCouplingLearnTactileFeedback object
    VectorNN_N* pmnn_output_vector;             // (pointer to) the output vector

    /**
     * NON-REAL-TIME!!!
     */
    TCLearnObsAvoidFeatureParameter();

    /**
     * NON-REAL-TIME!!!
     */
    TCLearnObsAvoidFeatureParameter(RealTimeAssertor* real_time_assertor,
                                    CanonicalSystemDiscrete* canonical_system_discrete,
                                    uint input_data_file_format,
                                    uint AF_H14_feat_method,
                                    uint Num_AF_H14_beta_phi1_phi2_grid,
                                    uint Num_AF_H14_k_phi1_phi2_grid,
                                    double init_AF_H14_beta_phi1_phi2_low, double init_AF_H14_beta_phi1_phi2_high,
                                    double init_AF_H14_k_phi1_phi2_low, double init_AF_H14_k_phi1_phi2_high,
                                    uint Num_AF_H14_k_phi3_grid=0,
                                    double init_AF_H14_k_phi3_low=0.0, double init_AF_H14_k_phi3_high=0.0,
                                    uint init_distance_kernel_scale_grid_mode=_LINEAR_,
                                    uint PF_DYN2_feat_method=_PF_DYN2_NO_USE_,
                                    uint Num_PF_DYN2_beta_grid=0,
                                    uint Num_PF_DYN2_k_grid=0,
                                    double init_PF_DYN2_beta_low=0.0, double init_PF_DYN2_beta_high=0.0,
                                    double init_PF_DYN2_k_low=0.0, double init_PF_DYN2_k_high=0.0,
                                    uint KGF_feat_method=_KGF_NO_USE_,
                                    uint Num_KGF_beta_grid=0,
                                    uint Num_KGF_k_grid=0,
                                    uint Num_KGF_s_grid=0,
                                    double init_KGF_beta_low=0.0, double init_KGF_beta_high=0.0,
                                    double init_KGF_k_low=0.0, double init_KGF_k_high=0.0,
                                    double init_KGF_s_low=0.0, double init_KGF_s_high=0.0,
                                    uint NN_feat_method=_NN_NO_USE_,
                                    uint Num_NN_input=0,
                                    uint Num_NN_hidden_layer_1=0,
                                    uint Num_NN_hidden_layer_2=0,
                                    uint Num_NN_output=0,
                                    const char* NN_params_directory_path="",
                                    VectorNN_N* pmnn_input_vector_ptr=nullptr,
                                    VectorNN_N* pmnn_phase_kernel_modulation_ptr=nullptr,
                                    VectorNN_N* pmnn_output_vector_ptr=nullptr);

    /**
     * Checks the validity of this data structure.
     *
     * @return Valid (true) or invalid (false)
     */
    bool isValid();
    
    ~TCLearnObsAvoidFeatureParameter();

};
}
#endif
