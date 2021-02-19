function [  ] = testTensorFlowTrainedPMNNFeedbackModel( task_type, learn_fb_task, ...
                                                        dmp_root_dir_path, ...
                                                        data_learn_fb_task_subdir_path, ...
                                                        python_learn_fb_task_TF_models_prefix_subdir_path, ...
                                                        start_prim_num, end_prim_num, ...
                                                        test_unroll_datafiles_prefix, ...
                                                        PMNN_name )
    data_root_dir_path          = [dmp_root_dir_path, 'data/'];
    matlab_root_dir_path        = [dmp_root_dir_path, 'matlab/'];
    python_root_dir_path        = [dmp_root_dir_path, 'python/'];

    data_learn_fb_task_dir_path                     = [data_root_dir_path, 'dmp_coupling/', data_learn_fb_task_subdir_path];
    data_learn_fb_task_PMNN_dir_path                = [data_learn_fb_task_dir_path, 'neural_nets/pmnn/'];
    data_learn_fb_task_PMNN_unroll_test_dir_path    = [data_learn_fb_task_PMNN_dir_path, 'unroll_test_dataset/'];
    data_learn_fb_task_PMNN_python_models_dir_path  = [data_learn_fb_task_PMNN_dir_path, 'python_models/'];

    python_learn_fb_task_dir_path           = [python_root_dir_path, 'dmp_coupling/', learn_fb_task, '/'];
    python_learn_fb_task_TF_models_dir_path = [python_learn_fb_task_dir_path, python_learn_fb_task_TF_models_prefix_subdir_path, 'models/'];

    addpath([matlab_root_dir_path, 'utilities/']);
    addpath([matlab_root_dir_path, 'neural_nets/feedforward/pmnn/']);

    reinit_selection_idx= dlmread([python_learn_fb_task_TF_models_dir_path, 'reinit_selection_idx.txt']);
    TF_max_train_iters  = dlmread([python_learn_fb_task_TF_models_dir_path, 'TF_max_train_iters.txt']);

    for np = start_prim_num:end_prim_num
        X               = dlmread([data_learn_fb_task_PMNN_unroll_test_dir_path, test_unroll_datafiles_prefix, 'test_unroll_prim_',num2str(np),'_X_raw_',task_type,'.txt']);
        normalized_phase_PSI_mult_phase_V   = dlmread([data_learn_fb_task_PMNN_unroll_test_dir_path, test_unroll_datafiles_prefix, 'test_unroll_prim_',num2str(np),'_normalized_phase_PSI_mult_phase_V_',task_type,'.txt']);
        Ct_target       = dlmread([data_learn_fb_task_PMNN_unroll_test_dir_path, test_unroll_datafiles_prefix, 'test_unroll_prim_',num2str(np),'_Ct_target_',task_type,'.txt']);

        T               = load([data_learn_fb_task_PMNN_python_models_dir_path, 'prim_', num2str(np), '_Ctt_test_prediction.mat']);
        Ctt_test_prediction_TF  = T.('Ctt_test_prediction');

        D_input             = size(X, 2);
        regular_NN_hidden_layer_topology = dlmread([python_learn_fb_task_TF_models_dir_path, 'regular_NN_hidden_layer_topology.txt']);
        N_phaseLWR_kernels  = size(normalized_phase_PSI_mult_phase_V, 2);
        D_output            = size(Ct_target, 2);

        regular_NN_hidden_layer_activation_func_list = readStringsToCell([python_learn_fb_task_TF_models_dir_path, 'regular_NN_hidden_layer_activation_func_list.txt']);

        NN_info.name                = PMNN_name;
        NN_info.topology            = [D_input, regular_NN_hidden_layer_topology, N_phaseLWR_kernels, D_output];
        NN_info.activation_func_list= {'identity', regular_NN_hidden_layer_activation_func_list{:}, 'identity', 'identity'};
        NN_info.filepath            = [data_learn_fb_task_PMNN_python_models_dir_path, 'prim_', num2str(np), '_params_reinit_',num2str(reinit_selection_idx(1,np)),'_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];

        [ Ctt_test_prediction_MATLAB, ~ ]   = performPMNNPrediction( NN_info, X, normalized_phase_PSI_mult_phase_V );

        diff_Ctt_test_prediction            = Ctt_test_prediction_MATLAB - Ctt_test_prediction_TF;
        L2_norm_diff_Ctt_test_prediction    = norm(diff_Ctt_test_prediction);
        max_abs_diff_Ctt_test_prediction    = max(max(abs(diff_Ctt_test_prediction)));
        rel_abs_diff_Ctt_test_prediction    = abs(diff_Ctt_test_prediction ./ Ctt_test_prediction_TF);
        max_rel_abs_diff_Ctt_test_prediction= max(max(rel_abs_diff_Ctt_test_prediction));
        disp(['L2_norm_diff_Ctt_test_prediction     = ', num2str(L2_norm_diff_Ctt_test_prediction)]);
        disp(['max_abs_diff_Ctt_test_prediction     = ', num2str(max_abs_diff_Ctt_test_prediction)]);
        disp(['max_rel_abs_diff_Ctt_test_prediction = ', num2str(max_rel_abs_diff_Ctt_test_prediction)]);
    end
end