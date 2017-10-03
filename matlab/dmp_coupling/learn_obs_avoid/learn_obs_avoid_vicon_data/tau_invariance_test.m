% Author: Giovanni Sutanto
% Date  : October 13, 2016

clear  	all;
close   all;
clc;

addpath('../utilities/');

[~, name] = system('hostname');
if (strcmp(name(1,1:end-1), 'gsutanto-ThinkPad-T430s') == 1)
    root_figure_path    = '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/';
else
    root_figure_path    = 'figures_multi/';
end

unrolling_param.is_comparing_with_cpp_implementation                = 0;

unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;

% learning_param.max_cond_number = 1000.0;   % Recommendation from Stefan (for Robotic applications)
learning_param.max_cond_number              = 5e3;
learning_param.feature_variance_threshold   = 1e-4;
learning_param.max_abs_ard_weight_threshold = 7.5e3;

if (unrolling_param.is_comparing_with_cpp_implementation)
    unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 0;
    unrolling_param.is_plot_unrolling   = 0;
    loa_feat_methods                    = 2;
    feat_constraint_mode                = '_CONSTRAINED_';
    learning_param.N_iter_ard          	= 1;
    selected_obs_avoid_setting_numbers  = [27, 119];
    max_num_trajs_per_setting           = 2;
    feature_threshold_mode              = 3;
    is_tau_invariant                    = [1];
    tau_multipliers                     = [1];
    
    figure_path                         = '';
else
    unrolling_param.is_plot_unrolling  	= 1;
    
    % possible loa_feat_method:
    % loa_feat_method == 0: Akshara's Humanoids'14 features (_CONSTRAINED_)
    % loa_feat_method == 1: Potential Field 2nd Dynamic Obst Avoid features
    % loa_feat_method == 2: Giovanni's Kernelized General Features v01
    % loa_feat_method == 3: Franzi's Kernelized General Features v01
    % loa_feat_method == 4: Potential Field 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
    % loa_feat_method == 5: Akshara's Humanoids'14 features (_UNCONSTRAINED_)
    % loa_feat_method == 6: Potential Field 4th Dynamic Obst Avoid features (have some sense of KGF too...)
    % loa_feat_method == 7: Neural Network
    % (enclose in a cell if you want to use multiple feature types, like in the following example below ...)
%     loa_feat_methods    = {2, 4};
    loa_feat_methods    = 7;
    
    % This feat_constraint_mode is for loa_feat_method 0, 1, 2, and 4.
    % For loa_feat_method 3 and 5, this value setting for
    % feat_constraint_mode is ignored,
    % and overridden with feat_constraint_mode = '_UNCONSTRAINED_'.
    % (in utilities/initializeAllInvolvedLOAparams.m
    %  or utilities/initializeFM_KGFv01LearnObsAvoidFeatParam.m)
    feat_constraint_mode                = '_CONSTRAINED_';
    
    learning_param.N_iter_ard        	= 200;

    selected_obs_avoid_setting_numbers  = [194];
    max_num_trajs_per_setting           = 500;
    feature_threshold_mode              = 7;
    is_tau_invariant                    = [1, 0];
    tau_multipliers                     = [1, 3, 7];

    figure_path     = root_figure_path;
    if (exist([figure_path, '/tau_invariance_testing/'], 'dir') ~= 7) % if directory NOT exist
        mkdir(figure_path, 'tau_invariance_testing'); % create directory
    end
    figure_path     = [figure_path, '/tau_invariance_testing/'];
end

diary([figure_path, 'tau_invariance_test_log.txt']);
diary on;

loa_feat_method_IDs     = '';
if (iscell(loa_feat_methods) == 1)
    for fmidx = 1:length(loa_feat_methods)
        loa_feat_method_IDs = [loa_feat_method_IDs,num2str(loa_feat_methods{1,fmidx}),'_'];
    end
else
    loa_feat_method_IDs   	= [loa_feat_method_IDs,num2str(loa_feat_methods),'_'];
end

unrolling_param.verify_NN_inference    	= 0;

learning_param.learning_constraint_mode = '_NONE_';

D               = 3;

n_rfs           = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order         = 1;    % DMP is using 2nd order canonical system

%% Data Conversion/Loading

data_filepath                       = '../data/data_multi_demo_vicon_static_global_coord.mat';
unrolling_param.cpp_impl_dump_path  = '../../../../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/multi_demo_vicon/';

% if input file is not exist yet, then create one (convert from C++ textfile format):
if (~exist(data_filepath, 'file'))
    convert_loa_vicon_data_to_mat_format;
end

load(data_filepath);

% end of Data Creation/Loading

%% Baseline Primitive Learning
disp('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...');
[ cart_coord_dmp_baseline_params, ...
  unrolling_param.cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord.baseline, data_global_coord.dt, n_rfs, c_order);

% end of Baseline Primitive Learning

for is_tau_invariant_idx=1:size(is_tau_invariant, 2)
    rng(1234)
    
    if (is_tau_invariant(1, is_tau_invariant_idx) == 1)
        disp('Using Tau-Invariant Features ...');
    else
        disp('Using Tau-Variant Features ...');
    end

    %% Obstacle Avoidance Features Grid Setting

    loa_feat_param = initializeAllInvolvedLOAparams( loa_feat_methods, ...
                                                     cart_coord_dmp_baseline_params.c_order, ...
                                                     learning_param, ...
                                                     unrolling_param, ...
                                                     feat_constraint_mode, ...
                                                     is_tau_invariant(1, is_tau_invariant_idx) );

    % end of Obstacle Avoidance Features Grid Setting
    
    %% Computation of Obstacle Avoidance Features and Target Coupling Term

    disp('Computing Obstacle Avoidance Features and Target Coupling Term');
    N_total_settings                = length(selected_obs_avoid_setting_numbers);

    sub_X_setting_cell                      = cell(N_total_settings, 1);
    sub_Ct_target_setting_cell              = cell(N_total_settings, 1);
    if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
        sub_Ct_target_setting_1st_demo_cell = cell(N_total_settings, 1);
    end
    for i=1:N_total_settings
        setting_num             = selected_obs_avoid_setting_numbers(1, i);
        N_demo_this_setting   	= min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{setting_num,2}, 2));

        sub_X_demo_cell                 = cell(N_demo_this_setting, 1);
        sub_Ct_target_demo_cell         = cell(N_demo_this_setting, 1);
        for j=1:N_demo_this_setting
            tic
            disp(['   Setting #', num2str(setting_num), ...
                  ' (', num2str(i), '/', num2str(N_total_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);
            [ sub_X_demo_cell{j,1}, ...
              sub_Ct_target_demo_cell{j,1} ]    = computeSubFeatMatAndSubTargetCt( data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                                   data_global_coord.obs_avoid{setting_num,1}, ...
                                                                                   data_global_coord.dt, ...
                                                                                   cart_coord_dmp_baseline_params, ...
                                                                                   loa_feat_methods, ...
                                                                                   loa_feat_param );
            toc
        end
        sub_X_setting_cell{i,1}                 = cell2mat(sub_X_demo_cell);
        sub_Ct_target_setting_cell{i,1}         = cell2mat(sub_Ct_target_demo_cell);
        if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
            sub_Ct_target_setting_1st_demo_cell{i,1}    = sub_Ct_target_demo_cell{1,1};
        end
    end
    X                           = cell2mat(sub_X_setting_cell);
    Ct_target                   = cell2mat(sub_Ct_target_setting_cell);
    if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
        Ct_target_all_1st_demo  = cell2mat(sub_Ct_target_setting_1st_demo_cell);
    end

    learning_param.retain_idx           = [1:size(X,2)];

    if (unrolling_param.is_comparing_with_cpp_implementation)
        X_cpp_impl          = dlmread([unrolling_param.cpp_impl_dump_path, 'X.txt']);
        Ct_target_cpp_impl  = dlmread([unrolling_param.cpp_impl_dump_path, 'Ct_target.txt']);

        mse_diff_X_matlab_vs_cpp_impl           = mean(mean( (X-X_cpp_impl).^2 ));
        mse_diff_Ct_target_matlab_vs_cpp_impl   = mean(mean( (Ct_target-Ct_target_cpp_impl).^2 ));

        disp(['MSE diff X         on MATLAB vs C++ implementation = ', num2str(mse_diff_X_matlab_vs_cpp_impl)]);
        disp(['MSE diff Ct_target on MATLAB vs C++ implementation = ', num2str(mse_diff_Ct_target_matlab_vs_cpp_impl)]);
    end

    save([loa_feat_method_IDs, 'X.mat'], 'X');
    save([loa_feat_method_IDs, 'Ct_target.mat'], 'Ct_target');
    save([loa_feat_method_IDs, 'loa_feat_param.mat'], 'loa_feat_param');

    % end of Computation of Obstacle Avoidance Features and Target Coupling Term

    %% Feature Matrix Pre-pruning

    if (loa_feat_methods ~= 7)
        tic
        disp(['Pre-pruning Feature Matrix:']);
        retained_feature_idx    = prepruneFeatureMatrix(X, feature_threshold_mode);
        toc
    end

    % end of Feature Matrix Pre-pruning

    if (loa_feat_methods == 7)
        %% Regression with Neural Network

        curr_best_nmse_learn_test  	= inf;
        % For running multiple training trials
        NN_N_training_trials        = 5;
        NMSE_threshold              = 0.1;
        N_data_threshold            = 50000;
        proportion_test_set         = 0.2;
        for j = 1:NN_N_training_trials
            disp(['Training Trial #', num2str(j), '/', num2str(NN_N_training_trials)]);
            tic            
            % create the net
            net = fitnet([loa_feat_param.NN_net_struct]);
            net.layers{1:end-1}.transferFcn = 'poslin';
            net.layers{end}.transferFcn     = 'tansig';
            % randomly draw test data
            N_data          = size(X,1);
            if (N_data < N_data_threshold)
                % Taking concatenated 20% of data for testing
                test_idx    = randperm(N_data, round(proportion_test_set * N_data));
            else
                test_idx    = randperm(N_data, round(proportion_test_set * N_data_threshold));
            end
            train_idx       = setdiff([1:N_data], test_idx);

            X_train        	= X(train_idx,:);
            Ct_train       	= Ct_target(train_idx,:);

            X_test        	= X(test_idx,:);
            Ct_test       	= Ct_target(test_idx,:);

            if (j == 1)
                % some print-outs to make sure things are set properly:
                disp(['N_data_all   = ', num2str(N_data)]);
                disp(['N_data_train = ', num2str(size(X_train,1))]);
                disp(['N_data_test  = ', num2str(size(X_test,1))]);
            end

            % train net
            [net, ~]        = train(net,X_train',Ct_train');

            % predict on training data
            Ct_fit_train    = net(X_train');
            [~, nmse_learn_train]   = computeNMSE(Ct_fit_train', Ct_train);

            % predict on test data
            Ct_fit_test     = net(X_test');
            [~, nmse_learn_test]    = computeNMSE(Ct_fit_test', Ct_test);

            disp(['nmse_learn_train = ', num2str(nmse_learn_train)]);
            disp(['nmse_learn_test  = ', num2str(nmse_learn_test)]);

            if ((max(nmse_learn_train(1,2:3)) < NMSE_threshold) && ...
                (max(nmse_learn_test(1,2:3)) < NMSE_threshold))
                net_best    = net;
                break;
            elseif (max(nmse_learn_test(1,2:3)) < curr_best_nmse_learn_test)
                net_best    = net;
                curr_best_nmse_learn_test   = max(nmse_learn_test(1,2:3));
            end

            % gsutanto: I want to store the history of trained NN, it might
            %           be useful for me.
            if (exist('trained_NN_history', 'dir') ~= 7)    % if directory NOT exist
                mkdir('./', 'trained_NN_history');          % create directory
            end
            save(['./trained_NN_history/trained_NN_',num2str(j),'.mat'], 'net');

            toc
        end
    %         load net_multi.mat
        net = net_best;
        % compute prediction on all data (so that it's comparable to other
        % methods) and compute mse and nmse
        Ct_fit                          = net(X_train');
        Ct_learn_test                   = net(X_test');
        [mse_learning, nmse_learning] 	= computeNMSE(Ct_fit', Ct_train);
        [~, nmse_learn_test]            = computeNMSE(Ct_learn_test', Ct_test);

        disp(['nmse_learn_train = ', num2str(nmse_learning)]);
        disp(['nmse_learn_test  = ', num2str(nmse_learn_test)]);

        learning_param.net = net;

        % end of Regression with Neural Network

    else
        %% Regression with ARD

        tic
        disp(['Performing ARD:']);
        if (strcmp(learning_param.learning_constraint_mode, '_NONE_') == 1)
            [ w_ard, nmse_learning_ard, Ct_fit_ard, mse_learning_ard ] = learnUsingARD( X, Ct_target, learning_param.max_abs_ard_weight_threshold, learning_param.N_iter_ard, retained_feature_idx );
        elseif (strcmp(learning_param.learning_constraint_mode, '_PER_AXIS_') == 1)
            [ w_ard, nmse_learning_ard, Ct_fit_ard, mse_learning_ard ] = learnUsingARDperAxis( X, Ct_target, learning_param.max_abs_ard_weight_threshold, learning_param.N_iter_ard, retained_feature_idx );
        end
        toc

        loa_feat_param.w                        = w_ard;

        disp(['mse_learning             = ', num2str(mse_learning_ard)]);
        disp(['nmse_learning            = ', num2str(nmse_learning_ard)]);

        Ct_fit                                  = Ct_fit_ard;
        nmse_learning                           = nmse_learning_ard;
        mse_learning                            = mse_learning_ard;

        if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
            Ct_target_3D               	= reshape(Ct_target,3,size(Ct_target,1)/3).';
            Ct_fit_3D               	= reshape(Ct_fit,3,size(Ct_fit,1)/3).';
            [ mse_learning_3D, nmse_learning_3D ]   = computeNMSE( Ct_fit_3D, Ct_target_3D );
            disp(['mse_learning_per_dim     = ', num2str(mse_learning_3D)]);
            disp(['nmse_learning_per_dim    = ', num2str(nmse_learning_3D)]);
        end

        figure;
        for d=1:size(w_ard,2)
            subplot(size(w_ard,2),1,d);
            title(['w (with ARD) d=',num2str(d)]);
            hold on;
                plot(w_ard(:,d));
                legend('ARD');
            hold off;
        end

        save([loa_feat_method_IDs, 'w_ard.mat'], 'w_ard');

        % end of Regression with ARD
    end

    for tau_idx=1:size(tau_multipliers, 2)
        unrolling_param.tau_multiplier  = tau_multipliers(1, tau_idx);
        
        %% Unrolling on Trained Obstacle Settings

        disp(['Unrolling on Trained Obstacle Settings, with Movement Duration (Tau) ', num2str(unrolling_param.tau_multiplier), 'X Learning Tau']);
        sub_Ct_unroll_setting_cell              = cell(N_total_settings, 1);
        global_traj_unroll_setting_cell         = cell(N_total_settings, 1);

        for i=1:N_total_settings
            setting_num             = selected_obs_avoid_setting_numbers(1, i);
            if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
                N_demo_this_setting = 1;
            else
                N_demo_this_setting = min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{setting_num,2}, 2));
            end

            sub_Ct_unroll_demo_cell             = cell(N_demo_this_setting, 1);
            global_traj_unroll_setting_cell{i,1}= cell(3, N_demo_this_setting);
            for j=1:N_demo_this_setting
                tic
                disp(['   Setting #', num2str(setting_num), ...
                      ' (', num2str(i), '/', num2str(N_total_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);

                if (j == 1)
                    is_measuring_demo_performance_metric    = 1;
                else
                    is_measuring_demo_performance_metric    = 0;
                end

                [ sub_Ct_unroll_demo_cell{j,1}, ...
                  global_traj_unroll_setting_cell{i,1}(1:3,j), ...
                  ~ ] = unrollObsAvoidViconTraj(data_global_coord.baseline(:,1), ...
                                                data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                data_global_coord.obs_avoid{setting_num,1}, ...
                                                data_global_coord.dt, ...
                                                cart_coord_dmp_baseline_params, ...
                                                loa_feat_methods, ...
                                                loa_feat_param, ...
                                                learning_param, ...
                                                unrolling_param, ...
                                                0, 0);
                toc
            end
            sub_Ct_unroll_setting_cell{i,1}	= cell2mat(sub_Ct_unroll_demo_cell);

            if ((unrolling_param.is_plot_unrolling) || (strcmp(figure_path,'') ~= 1))
                figure_name                 = '';
                if (strcmp(figure_path,'') ~= 1)
                    end_plotting_idx        = N_total_settings;
                    figure_name             = [figure_name,'tau_'];
                    if (is_tau_invariant(1, is_tau_invariant_idx) == 1)
                        figure_name         = [figure_name,'in'];
                    end
                    figure_name             = [figure_name,'variant_'];
                    if (iscell(loa_feat_methods) == 1)
                        for fmidx = 1:length(loa_feat_methods)
                            figure_name     = [figure_name,num2str(loa_feat_methods{1,fmidx}),'_'];
                        end
                    else
                        figure_name         = [figure_name,num2str(loa_feat_methods),'_'];
                    end
                    figure_name             = [figure_name,'setting_',num2str(setting_num),'_with_',num2str(unrolling_param.tau_multiplier),'X_tau_multiplier.fig'];
                else
                    end_plotting_idx        = 10;
                end
                if (i <= end_plotting_idx)
                    if (strcmp(figure_path,'') ~= 1)
                        visualizeSetting( setting_num, 1, [figure_path,figure_name], 0, ...
                                          unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
                                          global_traj_unroll_setting_cell{i,1}(1:3,1));
                    else
                        visualizeSetting( setting_num, 1, 0, 0, ...
                                          unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
                                          global_traj_unroll_setting_cell{i,1}(1:3,1));
                    end
                end
            end
        end

        Ct_unroll           = cell2mat(sub_Ct_unroll_setting_cell);

        if (unrolling_param.is_comparing_with_cpp_implementation)
            Ct_unroll_cpp_impl  = dlmread([unrolling_param.cpp_impl_dump_path, 'Ct_unroll.txt']);

            mse_diff_Ct_unroll_matlab_vs_cpp_impl   = mean(mean( (Ct_unroll-Ct_unroll_cpp_impl).^2 ));
            disp(['MSE diff Ct_unroll on MATLAB vs C++ implementation = ', num2str(mse_diff_Ct_unroll_matlab_vs_cpp_impl)]);
        end

        % end of Unrolling on Trained Obstacle Settings
    end
end

diary off;