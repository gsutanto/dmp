function [ unrolling_param, ...
           learning_param, ...
           loa_feat_methods_to_be_evaluated, ...
           feat_constraint_mode, ...
           loa_feat_methods, ...
           max_num_trajs_per_setting, ...
           D, n_rfs, c_order ] = getConfigParams(  )
    unrolling_param.is_comparing_with_cpp_implementation                = 0;
    unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 0;
    unrolling_param.is_plot_unrolling                                   = 0;
    unrolling_param.verify_NN_inference                                 = 0;

    learning_param.max_cond_number              = 5e3;
    learning_param.feature_variance_threshold   = 1e-4;
    learning_param.max_abs_ard_weight_threshold = 7.5e3;
    learning_param.N_iter_ard                   = 200;
    learning_param.learning_constraint_mode     = '_NONE_';

    loa_feat_methods_to_be_evaluated= [7];

    feat_constraint_mode            = '_CONSTRAINED_';

    loa_feat_methods                = 7;    % Neural Network

    max_num_trajs_per_setting       = 500;

    D     	= 3;

    n_rfs  	= 25;   % Number of basis functions used to represent the forcing term of DMP
    c_order = 1;    % DMP is using 2nd order canonical system
end