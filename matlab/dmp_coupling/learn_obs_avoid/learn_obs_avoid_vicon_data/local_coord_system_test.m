% Author: Giovanni Sutanto
% Date  : August 01, 2016

clear  	all;
close   all;
clc;

rng(1234)

addpath('../utilities/');

[~, name] = system('hostname');
if (strcmp(name(1,1:end-1), 'gsutanto-ThinkPad-T430s') == 1)
    root_figure_path    = '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/';
else
    root_figure_path    = 'figures_multi/';
end

unrolling_param.is_comparing_with_cpp_implementation                = 0;

unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;

local_z_axis_rotation_angle_perturbation        = pi;   % 180.0 degree rotation

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
    ctraj_local_coordinate_frames       = 1;
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
    loa_feat_methods    = 3;
    
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
    ctraj_local_coordinate_frames       = [2,0];
end

figure_path                         = '';

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

for lc_idx=1:size(ctraj_local_coordinate_frames, 2)
    %% Baseline Primitive Learning

    disp('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...');
    [ cart_coord_dmp_baseline_params, ...
      unrolling_param.cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord.baseline, data_global_coord.dt, n_rfs, c_order, ctraj_local_coordinate_frames(1,lc_idx));

    % end of Baseline Primitive Learning

    %% Computation of Perturbed Unrolling Setting

    if (unrolling_param.is_comparing_with_cpp_implementation == 0)
        if (ctraj_local_coordinate_frames(1,lc_idx) ~= 0)
            anchor_T_global_to_local_H  = cart_coord_dmp_baseline_params.T_global_to_local_H;
            anchor_T_local_to_global_H  = cart_coord_dmp_baseline_params.T_local_to_global_H;
            anchor_mean_start_local     = cart_coord_dmp_baseline_params.mean_start_local;
            anchor_mean_goal_local      = cart_coord_dmp_baseline_params.mean_goal_local;
        end
        
        considered_baseline_traj_global                 = data_global_coord.baseline(:,1);
        considered_demo_obs_avoid_traj_global           = data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,2}(:,1);
        considered_point_obstacles_cart_position_global = data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,1};

        [cart_coord_dmp_perturbed_params]   = computePerturbedObsAvoidSetting(anchor_T_global_to_local_H, ...
                                                                              anchor_T_local_to_global_H, ...
                                                                              anchor_mean_start_local, ...
                                                                              anchor_mean_goal_local, ...
                                                                              considered_baseline_traj_global, ...
                                                                              considered_demo_obs_avoid_traj_global, ...
                                                                              considered_point_obstacles_cart_position_global, ...
                                                                              cart_coord_dmp_baseline_params, ...
                                                                              local_z_axis_rotation_angle_perturbation, ...
                                                                              ctraj_local_coordinate_frames(1,lc_idx));
    end

    % end of Computation of Perturbed Unrolling Setting

    %% Obstacle Avoidance Features Grid Setting

    loa_feat_param = initializeAllInvolvedLOAparams( loa_feat_methods, ...
                                                     cart_coord_dmp_baseline_params.c_order, ...
                                                     learning_param, ...
                                                     unrolling_param, ...
                                                     feat_constraint_mode );

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
        Ct_fit = net(X_train');
        Ct_learn_test = net(X_test');
        [mse_learning, nmse_learning]            = computeNMSE(Ct_fit', Ct_train);
        [~, nmse_learn_test]                     = computeNMSE(Ct_learn_test', Ct_test);

        disp(['nmse_learn_train = ', num2str(nmse_learning)]);
        disp(['nmse_learn_test = ', num2str(nmse_learn_test)]);

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

    %% Unrolling on Trained Obstacle Settings

    disp('Unrolling on Trained Obstacle Settings');
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
                if (iscell(loa_feat_methods) == 1)
                    for fmidx = 1:length(loa_feat_methods)
                        figure_name     = [figure_name,num2str(loa_feat_methods{1,fmidx}),'_'];
                    end
                else
                    figure_name         = [figure_name,num2str(loa_feat_methods),'_'];
                end
                figure_name             = [figure_name,'setting_',num2str(setting_num),'.fig'];
            else
                end_plotting_idx        = 10;
            end
            if (i <= end_plotting_idx)
                if (strcmp(figure_path,'') ~= 1)
                    if (exist([figure_path, '/trained_settings/'], 'dir') ~= 7) % if directory NOT exist
                        mkdir(figure_path, 'trained_settings'); % create directory
                    end
%                     visualizeSetting( setting_num, 1, [figure_path,'/trained_settings/',figure_name], 0, ...
%                                       unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
%                                       global_traj_unroll_setting_cell{i,1}(1:3,1));
                else
%                     visualizeSetting( setting_num, 1, 0, 0, ...
%                                       unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
%                                       global_traj_unroll_setting_cell{i,1}(1:3,1));
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

    if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
        Ct_unroll_aggregated        = reshape(Ct_unroll.', size(Ct_unroll,1)*size(Ct_unroll,2), 1);

        if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
            [ mse_unroll, nmse_unroll ] = computeNMSE( Ct_unroll_aggregated, Ct_target_all_1st_demo );
        else
            [ mse_unroll, nmse_unroll ] = computeNMSE( Ct_unroll_aggregated, Ct_target );
        end

        Ct_unroll_3D                = Ct_unroll;
        if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
            Ct_target_3D          	= reshape(Ct_target_all_1st_demo,3,size(Ct_target_all_1st_demo,1)/3).';
        else
            Ct_target_3D          	= reshape(Ct_target,3,size(Ct_target,1)/3).';
        end
        [ mse_unroll_3D, nmse_unroll_3D ]   = computeNMSE( Ct_unroll_3D, Ct_target_3D );
    else
        if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
            [ mse_unroll, nmse_unroll ] = computeNMSE( Ct_unroll, Ct_target_all_1st_demo );
        else
            Ct_unroll = Ct_target;
            [ mse_unroll, nmse_unroll ] = computeNMSE( Ct_unroll, Ct_target );
        end
    end

    disp(['mse_unroll           = ', num2str(mse_unroll)]);
    disp(['nmse_unroll          = ', num2str(nmse_unroll)]);

    if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
        disp(['mse_unroll_per_dim   = ', num2str(mse_unroll_3D)]);
        disp(['nmse_unroll_per_dim  = ', num2str(nmse_unroll_3D)]);
    end

    % end of Unrolling on Trained Obstacle Settings

    if (unrolling_param.is_comparing_with_cpp_implementation == 0)
        %% Unrolling on Perturbed Trained Obstacle Settings
    
        disp('Unrolling on Perturbed Trained Obstacle Settings');
        sub_Ct_perturb_setting_cell              = cell(N_total_settings, 1);
        global_traj_perturb_setting_cell         = cell(N_total_settings, 1);

        for i=1:N_total_settings
            setting_num             = selected_obs_avoid_setting_numbers(1, i);
            if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
                N_demo_this_setting = 1;
            else
                N_demo_this_setting = min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{setting_num,2}, 2));
            end

            sub_Ct_perturb_demo_cell             = cell(N_demo_this_setting, 1);
            global_traj_perturb_setting_cell{i,1}= cell(3, N_demo_this_setting);
            for j=1:N_demo_this_setting
                tic
                disp(['   Setting #', num2str(setting_num), ...
                      ' (', num2str(i), '/', num2str(N_total_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);

                if (j == 1)
                    is_measuring_demo_performance_metric    = 1;
                else
                    is_measuring_demo_performance_metric    = 0;
                end

                [ sub_Ct_perturb_demo_cell{j,1}, ...
                  global_traj_perturb_setting_cell{i,1}(1:3,j), ...
                  ~ ] = unrollObsAvoidViconPerturbedTraj(data_global_coord.dt, ...
                                                         cart_coord_dmp_baseline_params, ...
                                                         loa_feat_methods, ...
                                                         loa_feat_param, ...
                                                         learning_param, ...
                                                         unrolling_param, ...
                                                         cart_coord_dmp_perturbed_params);
                toc
            end
            sub_Ct_perturb_setting_cell{i,1}             = cell2mat(sub_Ct_perturb_demo_cell);

            if ((unrolling_param.is_plot_unrolling) || (strcmp(figure_path,'') ~= 1))
                figure_name                 = '';
                if (strcmp(figure_path,'') ~= 1)
                    end_plotting_idx        = N_total_settings;
                    if (iscell(loa_feat_methods) == 1)
                        for fmidx = 1:length(loa_feat_methods)
                            figure_name     = [figure_name,num2str(loa_feat_methods{1,fmidx}),'_'];
                        end
                    else
                        figure_name         = [figure_name,num2str(loa_feat_methods),'_'];
                    end
                    figure_name             = [figure_name,'setting_',num2str(setting_num),'.fig'];
                else
                    end_plotting_idx        = 10;
                end
                if (i <= end_plotting_idx)
                    if (strcmp(figure_path,'') ~= 1)
                        if (exist([figure_path, '/perturbed_trained_settings/'], 'dir') ~= 7) % if directory NOT exist
                            mkdir(figure_path, 'perturbed_trained_settings'); % create directory
                        end
%                         visualizeSetting( setting_num, 1, [figure_path,'/perturbed_trained_settings/',figure_name], 0, ...
%                                           unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
%                                           global_traj_perturb_setting_cell{i,1}(1:3,1));
                    else
%                         visualizeSetting( setting_num, 1, 0, 0, ...
%                                           unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
%                                           global_traj_perturb_setting_cell{i,1}(1:3,1));
                    end
                end
            end
        end

        Ct_perturb                      = cell2mat(sub_Ct_perturb_setting_cell);

        if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
            Ct_perturb_aggregated       = reshape(Ct_perturb.', size(Ct_perturb,1)*size(Ct_perturb,2), 1);

            if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
                [ mse_perturb, nmse_perturb ] = computeNMSE( Ct_perturb_aggregated, Ct_target_all_1st_demo );
            else
                [ mse_perturb, nmse_perturb ] = computeNMSE( Ct_perturb_aggregated, Ct_target );
            end

            Ct_perturb_3D                = Ct_perturb;
            if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
                Ct_target_3D          	= reshape(Ct_target_all_1st_demo,3,size(Ct_target_all_1st_demo,1)/3).';
            else
                Ct_target_3D          	= reshape(Ct_target,3,size(Ct_target,1)/3).';
            end
            [ mse_perturb_3D, nmse_perturb_3D ]   = computeNMSE( Ct_perturb_3D, Ct_target_3D );
        else
            if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
                [ mse_perturb, nmse_perturb ] = computeNMSE( Ct_perturb, Ct_target_all_1st_demo );
            else
                Ct_perturb = Ct_target;
                [ mse_perturb, nmse_perturb ] = computeNMSE( Ct_perturb, Ct_target );
            end
        end

        disp(['mse_perturb           = ', num2str(mse_perturb)]);
        disp(['nmse_perturb          = ', num2str(nmse_perturb)]);

        if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
            disp(['mse_perturb_per_dim   = ', num2str(mse_perturb_3D)]);
            disp(['nmse_perturb_per_dim  = ', num2str(nmse_perturb_3D)]);
        end

        % end of Unrolling on Perturbed Trained Obstacle Settings
    
        %% Comparison Plotting
    
        vicon_marker_radius             = 15.0/2000.0;  % in meter
        critical_position_marker_radius = 30/1000.0;    % in meter

        global_x    = [1;0;0];
        global_y    = [0;1;0];
        global_z    = [0;0;1];

        local_x     = cart_coord_dmp_baseline_params.T_local_to_global_H(1:3,1);
        local_y     = cart_coord_dmp_baseline_params.T_local_to_global_H(1:3,2);
        local_z     = cart_coord_dmp_baseline_params.T_local_to_global_H(1:3,3);

        msg         = cart_coord_dmp_baseline_params.mean_start_global;

        if (lc_idx == 1)
            h_axis_disp = figure;
            axis        equal;
            hold        on;
%                 px1   	= quiver3(0,0,0,global_x(1,1),global_x(2,1),global_x(3,1),0,'r-.','LineWidth',3);
%                 py1    	= quiver3(0,0,0,global_y(1,1),global_y(2,1),global_y(3,1),0,'g-.','LineWidth',3);
%                 pz1    	= quiver3(0,0,0,global_z(1,1),global_z(2,1),global_z(3,1),0,'b-.','LineWidth',3);
                px2   	= quiver3(msg(1,1),msg(2,1),msg(3,1),local_x(1,1),local_x(2,1),local_x(3,1),0,'r','LineWidth',3);
                py2   	= quiver3(msg(1,1),msg(2,1),msg(3,1),local_y(1,1),local_y(2,1),local_y(3,1),0,'g','LineWidth',3);
                pz2   	= quiver3(msg(1,1),msg(2,1),msg(3,1),local_z(1,1),local_z(2,1),local_z(3,1),0,'b','LineWidth',3);
                for ib = 1:size(data_global_coord.baseline, 2)
                    pbs1= plot3(data_global_coord.baseline{1,ib}(:,1),...
                                data_global_coord.baseline{1,ib}(:,2),...
                                data_global_coord.baseline{1,ib}(:,3),...
                                'c');
                end
                for ioa = 1:size(data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,2}, 2)
                    poa1= plot3(data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,2}{1,ioa}(:,1),...
                                data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,2}{1,ioa}(:,2),...
                                data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,2}{1,ioa}(:,3),...
                                'm');
                end
                for op=1:size(data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,1},1)
                    plot_sphere(vicon_marker_radius,...
                                data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,1}(op,1),...
                                data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,1}(op,2),...
                                data_global_coord.obs_avoid{selected_obs_avoid_setting_numbers,1}(op,3));
                end

                xlabel('x');
                ylabel('y');
                zlabel('z');
        %         legend([px1, py1, pz1, px2, py2, pz2, pbs1, poa1], 'global x-axis', 'global y-axis', 'global z-axis', ...
                [hleg1, hobj1] =    legend([px2, py2, pz2, pbs1, poa1], ...
                                           'local x-axis', 'local y-axis', 'local z-axis', ...
                                           'demo: baseline', 'demo: obstacle avoidance');
                textobj = findobj(hobj1, 'type', 'text');
                set(textobj, 'Interpreter', 'latex', 'fontsize', 30);
            hold        off;
        end

        h_3D        = figure;
        axis        equal;
        hold        on;
            pdbst  = plot3(considered_baseline_traj_global{1,1}(:,1),...
                           considered_baseline_traj_global{1,1}(:,2),...
                           considered_baseline_traj_global{1,1}(:,3),...
                           'c','LineWidth',3);
            pdoat  = plot3(considered_demo_obs_avoid_traj_global{1,1}(:,1),...
                           considered_demo_obs_avoid_traj_global{1,1}(:,2),...
                           considered_demo_obs_avoid_traj_global{1,1}(:,3),...
                           'm','LineWidth',3);
            puoat  = plot3(global_traj_unroll_setting_cell{1,1}{1,1}(:,1),...
                           global_traj_unroll_setting_cell{1,1}{1,1}(:,2),...
                           global_traj_unroll_setting_cell{1,1}{1,1}(:,3),...
                           'r','LineWidth',3);
            puoapt  = plot3(global_traj_perturb_setting_cell{1,1}{1,1}(:,1),...
                            global_traj_perturb_setting_cell{1,1}{1,1}(:,2),...
                            global_traj_perturb_setting_cell{1,1}{1,1}(:,3),...
                            'r-.','LineWidth',3);

            for op=1:size(considered_point_obstacles_cart_position_global,1)
                plot_sphere(vicon_marker_radius,...
                            considered_point_obstacles_cart_position_global(op,1),...
                            considered_point_obstacles_cart_position_global(op,2),...
                            considered_point_obstacles_cart_position_global(op,3));
                plot_sphere(vicon_marker_radius,...
                            cart_coord_dmp_perturbed_params.perturbed_point_obstacles_cart_position_global(op,1),...
                            cart_coord_dmp_perturbed_params.perturbed_point_obstacles_cart_position_global(op,2),...
                            cart_coord_dmp_perturbed_params.perturbed_point_obstacles_cart_position_global(op,3));
            end

%             if (ctraj_local_coordinate_frames(1,lc_idx) == 0)
%                 title('not using local coordinate transformation');
%             else
%                 title('using local coordinate transformation');
%             end
            xlabel('x');
            ylabel('y');
            [hleg2, hobj2] =    legend([pdbst, pdoat, puoat, puoapt], ...
                                       'demo: baseline', 'demo: obstacle avoidance', 'unroll: obs avoid (learned setting)', 'unroll: obs avoid (rotated)');
            textobj = findobj(hobj2, 'type', 'text');
            set(textobj, 'Interpreter', 'latex', 'fontsize', 30);
        hold        off;
        
        % end of Comparison Plotting
    end
    
    %% Output Argument Settings

%     performance_metric.nmse_learning        = nmse_learning;
%     performance_metric.mse_learning         = mse_learning;
%     performance_metric.nmse_unroll          = nmse_unroll;
%     performance_metric.mse_unroll           = mse_unroll;
%     if (unrolling_param.is_comparing_with_cpp_implementation == 0)
%         performance_metric.nmse_test        = nmse_learn_test;
%     end
% 
%     if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
%         performance_metric.nmse_learning_3D = nmse_learning_3D;
%         performance_metric.mse_learning_3D  = mse_learning_3D;
%         performance_metric.nmse_unroll_3D   = nmse_unroll_3D;
%         performance_metric.mse_unroll_3D    = mse_unroll_3D;
%     end

    % end of Output Argument Settings
end