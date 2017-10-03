function [ varargout ] = learnAndUnrollObsAvoidViconDataSetting( varargin )
    % Author: Giovanni Sutanto
    % Date  : August 18, 2016
    
    close                           all;
    
    addpath('../utilities/');
    
    %% Function Input Argument Acquisition
    
    data_global_coord               = varargin{1};
    cart_coord_dmp_baseline_params  = varargin{2};
    loa_feat_methods                = varargin{3};
    feat_constraint_mode            = varargin{4};
    learning_param                  = varargin{5};
    unrolling_param                 = varargin{6};
    if (nargin > 6)
        selected_obs_avoid_setting_numbers  = varargin{7};
    else
        selected_obs_avoid_setting_numbers  = [1:size(data_global_coord.obs_avoid,1)];
    end
    if (nargin > 7)
        max_num_trajs_per_setting           = varargin{8};
    else
        max_num_trajs_per_setting           = 500;
    end
    if (nargin > 8)
        figure_path                         = varargin{9};
    else
        figure_path                         = '';
    end
    if (nargin > 9)
        is_unrolling_on_unseen_settings     = varargin{10};
    else
        is_unrolling_on_unseen_settings     = 0;
    end
    if (nargin > 10)
        feature_threshold_mode              = varargin{11};
    else
        feature_threshold_mode              = 6;
    end
    if (nargin > 11)
        debugging_X_and_Ct_mode             = varargin{12};
    else
        debugging_X_and_Ct_mode             = 0;
    end
    
    loa_feat_method_IDs     = '';
    if (iscell(loa_feat_methods) == 1)
        for fmidx = 1:length(loa_feat_methods)
            loa_feat_method_IDs = [loa_feat_method_IDs,num2str(loa_feat_methods{1,fmidx}),'_'];
        end
    else
        loa_feat_method_IDs   	= [loa_feat_method_IDs,num2str(loa_feat_methods),'_'];
    end
    
    % end of Function Input Argument Acquisition

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
    if (debugging_X_and_Ct_mode > 0)
        sub_Ct_target_3D_setting_cell 	= cell(N_total_settings, 1);
        if (debugging_X_and_Ct_mode == 2)
            if (loa_feat_methods ~= 7)
                sub_X_train_setting_per_point_cell  = cell(N_total_settings, 1);
            else
                sub_X_train_setting_cell            = cell(N_total_settings, 1);
            end
        end
    end
    for i=1:N_total_settings
        setting_num             = selected_obs_avoid_setting_numbers(1, i);
        N_demo_this_setting   	= min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{setting_num,2}, 2));
        
        sub_X_demo_cell                 = cell(N_demo_this_setting, 1);
        sub_Ct_target_demo_cell         = cell(N_demo_this_setting, 1);
        if (debugging_X_and_Ct_mode > 0)
            sub_Ct_target_3D_demo_cell  = cell(N_demo_this_setting, 1);
            if (debugging_X_and_Ct_mode == 2)
                sub_X_demo_per_point_cell   = cell(N_demo_this_setting, 1);
            end
        end
        for j=1:N_demo_this_setting
            tic
            disp(['   Setting #', num2str(setting_num), ...
                  ' (', num2str(i), '/', num2str(N_total_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);
            if ((debugging_X_and_Ct_mode == 2) && (loa_feat_methods ~= 7))
                [ sub_X_demo_cell{j,1}, ...
                  sub_Ct_target_demo_cell{j,1}, ...
                  sub_Ct_target_3D_demo_cell{j,1}, ...
                  sub_X_demo_per_point_cell{j,1} ] = computeSubFeatMatAndSubTargetCt( data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                                      data_global_coord.obs_avoid{setting_num,1}, ...
                                                                                      data_global_coord.dt, ...
                                                                                      cart_coord_dmp_baseline_params, ...
                                                                                      loa_feat_methods, ...
                                                                                      loa_feat_param );
            elseif ((debugging_X_and_Ct_mode == 1) || ((debugging_X_and_Ct_mode == 2) && (loa_feat_methods == 7)))
                [ sub_X_demo_cell{j,1}, ...
                  sub_Ct_target_demo_cell{j,1}, ...
                  sub_Ct_target_3D_demo_cell{j,1} ] = computeSubFeatMatAndSubTargetCt( data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                                       data_global_coord.obs_avoid{setting_num,1}, ...
                                                                                       data_global_coord.dt, ...
                                                                                       cart_coord_dmp_baseline_params, ...
                                                                                       loa_feat_methods, ...
                                                                                       loa_feat_param );
            else
                [ sub_X_demo_cell{j,1}, ...
                  sub_Ct_target_demo_cell{j,1} ]    = computeSubFeatMatAndSubTargetCt( data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                                       data_global_coord.obs_avoid{setting_num,1}, ...
                                                                                       data_global_coord.dt, ...
                                                                                       cart_coord_dmp_baseline_params, ...
                                                                                       loa_feat_methods, ...
                                                                                       loa_feat_param );
            end
            toc
        end
        sub_X_setting_cell{i,1}                 = cell2mat(sub_X_demo_cell);
        sub_Ct_target_setting_cell{i,1}         = cell2mat(sub_Ct_target_demo_cell);
        if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
            sub_Ct_target_setting_1st_demo_cell{i,1}    = sub_Ct_target_demo_cell{1,1};
        end
        
        if (debugging_X_and_Ct_mode > 0)
            sub_Ct_target_3D_setting_cell{i,1}  = sub_Ct_target_3D_demo_cell;
            if (debugging_X_and_Ct_mode == 2)
                if (loa_feat_methods ~= 7)
                    sub_X_train_setting_per_point_cell{i,1} = sub_X_demo_per_point_cell;
                else
                    sub_X_train_setting_cell{i,1}           = sub_X_demo_cell;
                end
            end
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
%                 start_idx = randperm(floor(N_data*(1-proportion_test_set)), 1);
%                 test_idx = start_idx:start_idx+proportion_test_set*N_data;
                test_idx    = randperm(N_data, round(proportion_test_set * N_data));
            else
%                 start_idx = randperm(floor(N_data*(1-proportion_test_set)), 1);
%                 test_idx = start_idx:start_idx+proportion_test_set*N_data_threshold;
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

        % Currently debugging X and Ct is only working for one setting:
        if ((debugging_X_and_Ct_mode > 0) && (N_total_settings == 1))
            sub_Ct_fit_3D_setting_cell      = cell(1, 1);
            row_sizes                       = cellfun('size', sub_Ct_target_3D_setting_cell{1,1}, 1).';
            Ct_fit_all                      = net(X');
            sub_Ct_fit_3D_setting_cell{1,1} = mat2cell(Ct_fit_all.', row_sizes, [3]);
        end
        
        % end of Regression with Neural Network

    else
        %% Regression with ARD
        
        if (unrolling_param.is_comparing_with_cpp_implementation == 0)
            N_data_threshold = 50000;
            proportion_test_set = 0.2;
            % randomly draw test data
            N_data          = size(X,1);
            if (N_data < N_data_threshold)
                test_idx    = randperm(N_data, round(proportion_test_set * N_data));
            else
                test_idx    = randperm(N_data, round(proportion_test_set * N_data_threshold));
            end
            train_idx       = setdiff([1:N_data], test_idx);

            X_train        	= X(train_idx,:);
            Ct_train       	= Ct_target(train_idx,:);

            X_test        	= X(test_idx,:);
            Ct_test       	= Ct_target(test_idx,:);

            tic
            disp(['Performing ARD:']);
            if (strcmp(learning_param.learning_constraint_mode, '_NONE_') == 1)
                [ w_ard, nmse_learning_ard, Ct_fit_ard, mse_learning_ard ] = learnUsingARD( X_train, Ct_train, 1e16, learning_param.N_iter_ard, retained_feature_idx );
            elseif (strcmp(learning_param.learning_constraint_mode, '_PER_AXIS_') == 1)
                [ w_ard, nmse_learning_ard, Ct_fit_ard, mse_learning_ard ] = learnUsingARDperAxis( X_train, Ct_train, 1e16, learning_param.N_iter_ard, retained_feature_idx );
            end
            toc
        else
            tic
            disp(['Performing ARD:']);
            if (strcmp(learning_param.learning_constraint_mode, '_NONE_') == 1)
                [ w_ard, nmse_learning_ard, Ct_fit_ard, mse_learning_ard ] = learnUsingARD( X, Ct_target, learning_param.max_abs_ard_weight_threshold, learning_param.N_iter_ard, retained_feature_idx );
            elseif (strcmp(learning_param.learning_constraint_mode, '_PER_AXIS_') == 1)
                [ w_ard, nmse_learning_ard, Ct_fit_ard, mse_learning_ard ] = learnUsingARDperAxis( X, Ct_target, learning_param.max_abs_ard_weight_threshold, learning_param.N_iter_ard, retained_feature_idx );
            end
            toc
        end

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
        
        if (unrolling_param.is_comparing_with_cpp_implementation == 0)
            % predict on test data
            Ct_fit_test     =  X_test*w_ard;
            [~, nmse_learn_test]    = computeNMSE(Ct_fit_test, Ct_test);
            disp(['nmse_test            = ', num2str(nmse_learn_test)]);
        end

        % Currently debugging X and Ct is only working for one setting:
        if ((debugging_X_and_Ct_mode > 0) && (N_total_settings == 1))
            sub_Ct_fit_3D_setting_cell      = cell(1, 1);
            row_sizes                       = cellfun('size', sub_Ct_target_3D_setting_cell{1,1}, 1).';
            if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
                sub_Ct_fit_3D_setting_cell{1,1} = mat2cell(Ct_fit_3D, row_sizes, [3]);
            else
                sub_Ct_fit_3D_setting_cell{1,1} = mat2cell(Ct_fit, row_sizes, [3]);
            end
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
    
    normed_closest_distance_to_obs_traj_human_1st_demo_setting  = cell(N_total_settings, 1);
    normed_closest_distance_to_obs_human_1st_demo_setting       = zeros(N_total_settings, 1);
    normed_closest_distance_to_obs_human_1st_demo_setting_idx   = zeros(N_total_settings, 1);
    final_distance_to_goal_vector_human_1st_demo_setting        = zeros(N_total_settings, 1);
    
    normed_closest_distance_to_obs_traj_per_trained_setting_cell= cell(N_total_settings, 1);
    normed_closest_distance_to_obs_per_trained_setting_cell    	= cell(N_total_settings, 1);
    normed_closest_distance_to_obs_per_trained_setting_idx_cell = cell(N_total_settings, 1);
    final_distance_to_goal_vector_per_trained_setting_cell     	= cell(N_total_settings, 1);
    normed_closest_distance_to_obs_overall_train_demos_per_setting 	= zeros(N_total_settings, 1);
    normed_closest_distance_to_obs_overall_train_demos_per_sett_idx = zeros(N_total_settings, 1);
    worst_final_distance_to_goal_per_trained_setting        = zeros(N_total_settings, 1);
    worst_final_distance_to_goal_per_trained_setting_idx    = zeros(N_total_settings, 1);
    if (debugging_X_and_Ct_mode > 0)
        sub_Ct_unroll_setting_cell_cell   	= cell(N_total_settings, 1);
        if (debugging_X_and_Ct_mode == 2)
            sub_X_unroll_setting_cell       = cell(N_total_settings, 1);
        end
    end
    for i=1:N_total_settings
        setting_num             = selected_obs_avoid_setting_numbers(1, i);
        if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
            N_demo_this_setting = 1;
        else
            N_demo_this_setting = min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{setting_num,2}, 2));
        end
        
        sub_Ct_unroll_demo_cell             = cell(N_demo_this_setting, 1);
        global_traj_unroll_setting_cell{i,1}= cell(3, N_demo_this_setting);
        normed_closest_distance_to_obs_traj_per_trained_setting_cell{i,1}   = cell(N_demo_this_setting, 1);
        normed_closest_distance_to_obs_per_trained_setting_cell{i,1}    	= zeros(N_demo_this_setting, 1);
        normed_closest_distance_to_obs_per_trained_setting_idx_cell{i,1}    = zeros(N_demo_this_setting, 1);
        final_distance_to_goal_vector_per_trained_setting_cell{i,1}     	= zeros(N_demo_this_setting, 1);
        if (debugging_X_and_Ct_mode == 2)
            sub_X_unroll_demo_cell          = cell(N_demo_this_setting, 1);
        end
        for j=1:N_demo_this_setting
            tic
            disp(['   Setting #', num2str(setting_num), ...
                  ' (', num2str(i), '/', num2str(N_total_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);

            if (j == 1)
                is_measuring_demo_performance_metric    = 1;
            else
                is_measuring_demo_performance_metric    = 0;
            end
            
            if (debugging_X_and_Ct_mode == 2)
                [ sub_Ct_unroll_demo_cell{j,1}, ...
                  global_traj_unroll_setting_cell{i,1}(1:3,j), ...
                  ~, ...
                  buf_normed_closest_distance_to_obs_traj_human_1st_demo_setting, ...
                  buf_final_distance_to_goal_vector_human_1st_demo_setting, ...
                  base_normed_closest_distance_to_obs_traj_human_1st_demo_setting, ...
                  base_final_distance_to_goal_vector_human_1st_demo_setting, ...
                  normed_closest_distance_to_obs_traj_per_trained_setting_cell{i,1}{j,1}, ...
                  final_distance_to_goal_vector_per_trained_setting_cell{i,1}(j,1), ...
                  sub_X_unroll_demo_cell{j,1} ]   = unrollObsAvoidViconTraj( data_global_coord.baseline(:,1), ...
                                                                             data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                             data_global_coord.obs_avoid{setting_num,1}, ...
                                                                             data_global_coord.dt, ...
                                                                             cart_coord_dmp_baseline_params, ...
                                                                             loa_feat_methods, ...
                                                                             loa_feat_param, ...
                                                                             learning_param, ...
                                                                             unrolling_param, ...
                                                                             is_measuring_demo_performance_metric );
            else
                [ sub_Ct_unroll_demo_cell{j,1}, ...
                  global_traj_unroll_setting_cell{i,1}(1:3,j), ...
                  ~, ...
                  buf_normed_closest_distance_to_obs_traj_human_1st_demo_setting, ...
                  buf_final_distance_to_goal_vector_human_1st_demo_setting, ...
                  base_normed_closest_distance_to_obs_traj_human_1st_demo_setting, ...
                  base_final_distance_to_goal_vector_human_1st_demo_setting, ...
                  normed_closest_distance_to_obs_traj_per_trained_setting_cell{i,1}{j,1}, ...
                  final_distance_to_goal_vector_per_trained_setting_cell{i,1}(j,1) ] = unrollObsAvoidViconTraj( data_global_coord.baseline(:,1),...
                                                                                                                data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                                                                data_global_coord.obs_avoid{setting_num,1}, ...
                                                                                                                data_global_coord.dt, ...
                                                                                                                cart_coord_dmp_baseline_params, ...
                                                                                                                loa_feat_methods, ...
                                                                                                                loa_feat_param, ...
                                                                                                                learning_param, ...
                                                                                                                unrolling_param, ...
                                                                                                                is_measuring_demo_performance_metric );
            end
            if ((j == 1) && (isempty(buf_normed_closest_distance_to_obs_traj_human_1st_demo_setting) == 0) && ...
                (isnan(buf_final_distance_to_goal_vector_human_1st_demo_setting) == 0))
                
                normed_closest_distance_to_obs_traj_baseline_demo_setting{i,1}  = base_normed_closest_distance_to_obs_traj_human_1st_demo_setting;
                final_distance_to_goal_vector_baseline_demo_setting(i,1)        = base_final_distance_to_goal_vector_human_1st_demo_setting;

                normed_closest_distance_to_obs_traj_human_1st_demo_setting{i,1} = buf_normed_closest_distance_to_obs_traj_human_1st_demo_setting;
                final_distance_to_goal_vector_human_1st_demo_setting(i,1)       = buf_final_distance_to_goal_vector_human_1st_demo_setting;
            end
            
            [ normed_closest_distance_to_obs_per_trained_setting_cell{i,1}(j,1), ...
              normed_closest_distance_to_obs_per_trained_setting_idx_cell{i,1}(j,1) ]   = min(normed_closest_distance_to_obs_traj_per_trained_setting_cell{i,1}{j,1});
            
            toc
        end
        [normed_closest_distance_to_obs_baseline_demo_setting(i,1), ...
         normed_closest_distance_to_obs_baseline_demo_setting_idx(i,1)]    = min(normed_closest_distance_to_obs_traj_baseline_demo_setting{i,1});

        [normed_closest_distance_to_obs_human_1st_demo_setting(i,1), ...
         normed_closest_distance_to_obs_human_1st_demo_setting_idx(i,1)]    = min(normed_closest_distance_to_obs_traj_human_1st_demo_setting{i,1});

        [normed_closest_distance_to_obs_overall_train_demos_per_setting(i,1), ...
         normed_closest_distance_to_obs_overall_train_demos_per_sett_idx(i,1)]    = min(normed_closest_distance_to_obs_per_trained_setting_cell{i,1});
        [worst_final_distance_to_goal_per_trained_setting(i,1), ...
         worst_final_distance_to_goal_per_trained_setting_idx(i,1)] = max(final_distance_to_goal_vector_per_trained_setting_cell{i,1});
        sub_Ct_unroll_setting_cell{i,1}             = cell2mat(sub_Ct_unroll_demo_cell);
        if (debugging_X_and_Ct_mode > 0)
            sub_Ct_unroll_setting_cell_cell{i,1}   	= sub_Ct_unroll_demo_cell;
            if (debugging_X_and_Ct_mode == 2)
                sub_X_unroll_setting_cell{i,1}      = sub_X_unroll_demo_cell;
            end
        end

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
                    visualizeSetting( setting_num, 1, [figure_path,'/trained_settings/',figure_name], 0, ...
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
    
    
    [normed_closest_dist_to_obs_human_1st_demo_over_all_settings, ...
     normed_closest_dist_to_obs_human_1st_demo_over_all_settings_idx]   = min(normed_closest_distance_to_obs_human_1st_demo_setting);
    [worst_final_dist_to_goal_vector_human_1st_demo_overall_settings, ...
     worst_final_dist_to_goal_vector_human_1st_demo_overall_sett_idx]   = min(final_distance_to_goal_vector_human_1st_demo_setting);
    
    [normed_closest_distance_to_obs_over_all_trained_settings, ...
     normed_closest_distance_to_obs_over_all_trained_settings_idx]  = min(normed_closest_distance_to_obs_overall_train_demos_per_setting);
    [worst_final_distance_to_goal_over_all_trained_settings, ...
     worst_final_distance_to_goal_over_all_trained_setting_idx] = max(worst_final_distance_to_goal_per_trained_setting);
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
    
    %% Visualizing (Debugging) X and Ct
    
    if ((debugging_X_and_Ct_mode > 0) && (unrolling_param.is_comparing_with_cpp_implementation == 0))
        if (debugging_X_and_Ct_mode == 1)
            visualizeX_and_Ct( unrolling_param, ...
                               sub_Ct_target_3D_setting_cell, ...
                               sub_Ct_fit_3D_setting_cell, ...
                               sub_Ct_unroll_setting_cell_cell );
        elseif (debugging_X_and_Ct_mode == 2)
            if (loa_feat_methods ~= 7)
                visualizeX_and_Ct( unrolling_param, ...
                                   sub_Ct_target_3D_setting_cell, ...
                                   sub_Ct_fit_3D_setting_cell, ...
                                   sub_Ct_unroll_setting_cell_cell, ...
                                   sub_X_train_setting_per_point_cell, ...
                                   sub_X_unroll_setting_cell );
            else
                visualizeX_and_Ct( unrolling_param, ...
                                   sub_Ct_target_3D_setting_cell, ...
                                   sub_Ct_fit_3D_setting_cell, ...
                                   sub_Ct_unroll_setting_cell_cell, ...
                                   sub_X_train_setting_cell, ...
                                   sub_X_unroll_setting_cell );
            end
        end
    end
    
    % end of Visualizing (Debugging) X and Ct
    
    %% Unrolling on Unseen Obstacle Settings
    
    if (is_unrolling_on_unseen_settings)
        sphere_obs_center_coords    = generateSphereObstacleCenterCoordinates( cart_coord_dmp_baseline_params.mean_start_global, ...
                                                                               cart_coord_dmp_baseline_params.mean_goal_global );
        N_unseen_settings           = size(sphere_obs_center_coords, 2);
        sphere_params_global.radius = 0.05; % in meter
        disp('Unrolling on Unseen Obstacle Settings');
        sub_Ct_unroll_unseen_setting_cell           = cell(N_unseen_settings, 1);
        global_traj_unroll_unseen_setting_cell      = cell(N_unseen_settings, 1);
        local_traj_unroll_unseen_setting_cell       = cell(N_unseen_settings, 1);
        normed_closest_distance_to_obs_traj_per_unseen_setting_cell = cell(N_unseen_settings, 1);
        normed_closest_distance_to_obs_per_unseen_setting       = zeros(N_unseen_settings, 1);
        normed_closest_distance_to_obs_per_unseen_setting_idx   = zeros(N_unseen_settings, 1);
        final_distance_to_goal_vector_per_unseen_setting     	= zeros(N_unseen_settings, 1);
        point_obstacles_cart_position_global_cell               = cell(N_unseen_settings, 1);
        for i=1:N_unseen_settings
            sphere_params_global.center = sphere_obs_center_coords(:,i);
            tic
            disp(['   Unseen Setting #', num2str(i), '/', num2str(N_unseen_settings), ' ...']);
            [ sub_Ct_unroll_unseen_setting_cell{i,1}, ...
              global_traj_unroll_unseen_setting_cell{i,1}, ...
              local_traj_unroll_unseen_setting_cell{i,1}, ...
              normed_closest_distance_to_obs_traj_per_unseen_setting_cell{i,1}, ...
              final_distance_to_goal_vector_per_unseen_setting(i,1), ...
              point_obstacles_cart_position_global_cell{i,1} ]  = unrollObsAvoidSpherePointCloudsEvaluation( sphere_params_global, ...
                                                                                                             data_global_coord.dt, ...
                                                                                                             cart_coord_dmp_baseline_params, ...
                                                                                                             loa_feat_methods, ...
                                                                                                             loa_feat_param, ...
                                                                                                             learning_param );
            toc
            [normed_closest_distance_to_obs_per_unseen_setting(i,1),normed_closest_distance_to_obs_per_unseen_setting_idx(i,1)] = min(normed_closest_distance_to_obs_traj_per_unseen_setting_cell{i,1});
            
            if ((unrolling_param.is_plot_unrolling) || (strcmp(figure_path,'') ~= 1))
                figure_name                 = '';
                if (strcmp(figure_path,'') ~= 1)
                    figure_name             = [figure_name, loa_feat_method_IDs];
                    figure_name             = [figure_name,'unseen_setting_',num2str(i),'.fig'];
                    
                    if (exist([figure_path, '/unseen_settings/'], 'dir') ~= 7)  % if directory NOT exist
                        mkdir(figure_path, 'unseen_settings');  % create directory
                    end
                    visualizeSetting( 0, 1, [figure_path,'/unseen_settings/',figure_name], 0, ...
                                      unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
                                      global_traj_unroll_unseen_setting_cell{i,1}, ...
                                      point_obstacles_cart_position_global_cell{i,1} );
                else
                    visualizeSetting( 0, 1, 0, 0, ...
                                      unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
                                      global_traj_unroll_unseen_setting_cell{i,1}, ...
                                      point_obstacles_cart_position_global_cell{i,1} );
                end
            end
        end
        [normed_closest_distance_to_obs_over_all_unseen_settings, normed_closest_distance_to_obs_over_all_unseen_settings_idx] = min(normed_closest_distance_to_obs_per_unseen_setting);
        percentage_hitting_obstacle     = (length(find(normed_closest_distance_to_obs_per_unseen_setting <= 0))/length(normed_closest_distance_to_obs_per_unseen_setting)) * 100.0;
        [worst_final_distance_to_goal_over_all_unseen_settings, worst_final_distance_to_goal_over_all_unseen_settings_idx]  = max(final_distance_to_goal_vector_per_unseen_setting);
    end
    
    % end of Unrolling on Unseen Obstacle Settings
    
    %% Function Output Argument Settings
    
    performance_metric.nmse_learning        = nmse_learning;
    performance_metric.mse_learning         = mse_learning;
    performance_metric.nmse_unroll          = nmse_unroll;
    performance_metric.mse_unroll           = mse_unroll;
    if (unrolling_param.is_comparing_with_cpp_implementation == 0)
        performance_metric.nmse_test            = nmse_learn_test;
    end
        
    if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
        performance_metric.nmse_learning_3D = nmse_learning_3D;
        performance_metric.mse_learning_3D  = mse_learning_3D;
        performance_metric.nmse_unroll_3D   = nmse_unroll_3D;
        performance_metric.mse_unroll_3D    = mse_unroll_3D;
    end
    
    performance_metric.baseline.normed_closest_distance_to_obs_human_1st_demo_setting                   = normed_closest_distance_to_obs_baseline_demo_setting;
    performance_metric.baseline.final_distance_to_goal_vector_baseline_1st_demo_setting                 = final_distance_to_goal_vector_baseline_demo_setting;

    performance_metric.human_demo.normed_closest_distance_to_obs_traj_human_1st_demo_setting            = normed_closest_distance_to_obs_traj_human_1st_demo_setting;
    performance_metric.human_demo.normed_closest_distance_to_obs_human_1st_demo_setting                 = normed_closest_distance_to_obs_human_1st_demo_setting;
    performance_metric.human_demo.normed_closest_distance_to_obs_human_1st_demo_setting_idx             = normed_closest_distance_to_obs_human_1st_demo_setting_idx;
    performance_metric.human_demo.final_distance_to_goal_vector_human_1st_demo_setting                  = final_distance_to_goal_vector_human_1st_demo_setting;
    performance_metric.human_demo.normed_closest_dist_to_obs_human_1st_demo_over_all_settings           = normed_closest_dist_to_obs_human_1st_demo_over_all_settings;
    performance_metric.human_demo.normed_closest_dist_to_obs_human_1st_demo_over_all_settings_idx       = normed_closest_dist_to_obs_human_1st_demo_over_all_settings_idx;
    performance_metric.human_demo.worst_final_dist_to_goal_vector_human_1st_demo_overall_settings       = worst_final_dist_to_goal_vector_human_1st_demo_overall_settings;
    performance_metric.human_demo.worst_final_dist_to_goal_vector_human_1st_demo_overall_sett_idx       = worst_final_dist_to_goal_vector_human_1st_demo_overall_sett_idx;
    
    performance_metric.trained_settings.normed_closest_distance_to_obs_traj_per_trained_setting_cell 	= normed_closest_distance_to_obs_traj_per_trained_setting_cell;
    performance_metric.trained_settings.normed_closest_distance_to_obs_per_trained_setting_cell      	= normed_closest_distance_to_obs_per_trained_setting_cell;
    performance_metric.trained_settings.normed_closest_distance_to_obs_per_trained_setting_idx_cell    	= normed_closest_distance_to_obs_per_trained_setting_idx_cell;
    performance_metric.trained_settings.final_distance_to_goal_vector_per_trained_setting_cell        	= final_distance_to_goal_vector_per_trained_setting_cell;
    performance_metric.trained_settings.normed_closest_distance_to_obs_overall_train_demos_per_setting 	= normed_closest_distance_to_obs_overall_train_demos_per_setting;
    performance_metric.trained_settings.normed_closest_distance_to_obs_overall_train_demos_per_sett_idx = normed_closest_distance_to_obs_overall_train_demos_per_sett_idx;
    performance_metric.trained_settings.worst_final_distance_to_goal_per_trained_setting               	= worst_final_distance_to_goal_per_trained_setting;
    performance_metric.trained_settings.worst_final_distance_to_goal_per_trained_setting_idx           	= worst_final_distance_to_goal_per_trained_setting_idx;
    performance_metric.trained_settings.normed_closest_distance_to_obs_over_all_trained_settings       	= normed_closest_distance_to_obs_over_all_trained_settings;
    performance_metric.trained_settings.normed_closest_distance_to_obs_over_all_trained_settings_idx   	= normed_closest_distance_to_obs_over_all_trained_settings_idx;
    performance_metric.trained_settings.worst_final_distance_to_goal_over_all_trained_settings         	= worst_final_distance_to_goal_over_all_trained_settings;
    performance_metric.trained_settings.worst_final_distance_to_goal_over_all_trained_setting_idx      	= worst_final_distance_to_goal_over_all_trained_setting_idx;
    
    if (is_unrolling_on_unseen_settings)
        performance_metric.unseen_settings.normed_closest_distance_to_obs_over_all_unseen_settings      = normed_closest_distance_to_obs_over_all_unseen_settings;
        performance_metric.unseen_settings.normed_closest_distance_to_obs_over_all_unseen_settings_idx  = normed_closest_distance_to_obs_over_all_unseen_settings_idx;
        performance_metric.unseen_settings.normed_closest_distance_to_obs_per_unseen_setting            = normed_closest_distance_to_obs_per_unseen_setting;
        performance_metric.unseen_settings.normed_closest_distance_to_obs_per_unseen_setting_idx        = normed_closest_distance_to_obs_per_unseen_setting_idx;
        performance_metric.unseen_settings.normed_closest_distance_to_obs_traj_per_unseen_setting_cell  = normed_closest_distance_to_obs_traj_per_unseen_setting_cell;
        performance_metric.unseen_settings.percentage_hitting_obstacle                                  = percentage_hitting_obstacle;
        performance_metric.unseen_settings.worst_final_distance_to_goal_over_all_unseen_settings        = worst_final_distance_to_goal_over_all_unseen_settings;
        performance_metric.unseen_settings.worst_final_distance_to_goal_over_all_unseen_settings_idx  	= worst_final_distance_to_goal_over_all_unseen_settings_idx;
        performance_metric.unseen_settings.final_distance_to_goal_vector_per_unseen_setting             = final_distance_to_goal_vector_per_unseen_setting;
    end
    
    learning_unrolling_variables.X          = X;
    learning_unrolling_variables.Ct_target  = Ct_target;
    learning_unrolling_variables.Ct_fit     = Ct_fit;
    learning_unrolling_variables.Ct_unroll  = Ct_unroll;
    
    varargout(1)        = {performance_metric};
    if (nargout > 1)
        varargout(2)	= {learning_unrolling_variables};
    end
    if (nargout > 2)
        varargout(3)	= {learning_param};
    end
    
    % end of Function Output Argument Settings
end