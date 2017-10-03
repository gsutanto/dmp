function [ varargout ] = learnAndUnrollObsAvoidPF_DYN3ViconDataSetting( varargin )
    % Author: Giovanni Sutanto
    % Date  : August 18, 2016
    
    close                           all;
    
    addpath('../../utilities/');
    addpath('../');
    
    %% Function Input Argument Acquisition
    
    data_global_coord               = varargin{1};
    cart_coord_dmp_baseline_params  = varargin{2};
    loa_feat_methods                = varargin{3};
    feat_constraint_mode            = varargin{4};
    learning_param                  = varargin{5};
    unrolling_param                 = varargin{6};
    N_PF_DYN3_beta_grid             = varargin{7};
    PF_DYN3_beta_low                = varargin{8};
    PF_DYN3_beta_high               = varargin{9};
    N_PF_DYN3_k_grid                = varargin{10};
    PF_DYN3_k_low                   = varargin{11};
    PF_DYN3_k_high                  = varargin{12};
    N_PF_DYN3_s_grid                = varargin{13};
    PF_DYN3_s_low                   = varargin{14};
    PF_DYN3_s_high                  = varargin{15};
    PF_DYN3_beta_default_D          = varargin{16};
    PF_DYN3_s_default_D             = varargin{17};
    if (nargin > 17)
        selected_obs_avoid_setting_numbers  = varargin{18};
    else
        selected_obs_avoid_setting_numbers  = [1:size(data_global_coord.obs_avoid,1)];
    end
    if (nargin > 18)
        max_num_trajs_per_setting           = varargin{19};
    else
        max_num_trajs_per_setting           = 500;
    end
    if (nargin > 19)
        figure_path                         = varargin{20};
    else
        figure_path                         = '';
    end
    if (nargin > 20)
        is_unrolling_on_unseen_settings     = varargin{21};
    else
        is_unrolling_on_unseen_settings     = 0;
    end
    if (nargin > 21)
        feature_threshold_mode              = varargin{22};
    else
        feature_threshold_mode              = 6;
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
    
    loa_feat_param 	= initializeLearnObsAvoidFeatParam( cart_coord_dmp_baseline_params.c_order, ...
                                                        learning_param.learning_constraint_mode );
    loa_feat_param  = initializePF_DYN3LearnObsAvoidFeatParam( loa_feat_param, ...
                                                               N_PF_DYN3_beta_grid, ...
                                                               PF_DYN3_beta_low, ...
                                                               PF_DYN3_beta_high, ...
                                                               N_PF_DYN3_k_grid, ...
                                                               PF_DYN3_k_low, ...
                                                               PF_DYN3_k_high, ...
                                                               N_PF_DYN3_s_grid, ...
                                                               cart_coord_dmp_baseline_params.c_order, ...
                                                               feat_constraint_mode, ...
                                                               PF_DYN3_s_low, ...
                                                               PF_DYN3_s_high, ...
                                                               PF_DYN3_beta_default_D, ...
                                                               PF_DYN3_s_default_D );

    % end of Obstacle Avoidance Features Grid Setting
    
    %% Computation of Obstacle Avoidance Features and Target Coupling Term

    disp('Computing Obstacle Avoidance Features and Target Coupling Term');
    N_total_settings          	= length(selected_obs_avoid_setting_numbers);
    sub_X_setting_cell          = cell(N_total_settings, 1);
    sub_Ct_target_setting_cell 	= cell(N_total_settings, 1);
    for i=1:N_total_settings
        setting_num             = selected_obs_avoid_setting_numbers(1, i);
        N_demo_this_setting   	= min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{setting_num,2}, 2));
        sub_X_demo_cell        	= cell(N_demo_this_setting, 1);
        sub_Ct_target_demo_cell = cell(N_demo_this_setting, 1);
        for j=1:N_demo_this_setting
            tic
            disp(['   Setting #', num2str(setting_num), ...
                  ' (', num2str(i), '/', num2str(N_total_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);
            [ sub_X_demo_cell{j,1}, sub_Ct_target_demo_cell{j,1} ] = computeSubFeatMatAndSubTargetCt( data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                                                      data_global_coord.obs_avoid{setting_num,1}, ...
                                                                                                      data_global_coord.dt, ...
                                                                                                      cart_coord_dmp_baseline_params, ...
                                                                                                      loa_feat_methods, ...
                                                                                                      loa_feat_param );
            toc
        end
        sub_X_setting_cell{i,1}         = cell2mat(sub_X_demo_cell);
        sub_Ct_target_setting_cell{i,1}	= cell2mat(sub_Ct_target_demo_cell);
    end
    X                   = cell2mat(sub_X_setting_cell);
    Ct_target           = cell2mat(sub_Ct_target_setting_cell);

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
    
    retained_feature_idx    = [1:size(X,2)];

%     tic
%     disp(['Pre-pruning Feature Matrix:']);
%     retained_feature_idx    = prepruneFeatureMatrix(X, feature_threshold_mode);
%     toc
    
    % end of Feature Matrix Pre-pruning
    
    %% Regression (with ARD)

    if (loa_feat_methods ~= 7)
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
    end

    % end of Regression (with ARD)

    %% Unrolling on Trained Obstacle Settings

    disp('Unrolling on Trained Obstacle Settings');
    sub_Ct_unroll_setting_cell      = cell(N_total_settings, 1);
    global_traj_unroll_setting_cell = cell(N_total_settings, 1);
    for i=1:N_total_settings
        setting_num             = selected_obs_avoid_setting_numbers(1, i);
        N_demo_this_setting   	= min(max_num_trajs_per_setting, size(data_global_coord.obs_avoid{setting_num,2}, 2));
        sub_Ct_unroll_demo_cell = cell(N_demo_this_setting, 1);
        global_traj_unroll_setting_cell{i,1}= cell(3, N_demo_this_setting);
        for j=1:N_demo_this_setting
            tic
            disp(['   Setting #', num2str(setting_num), ...
                  ' (', num2str(i), '/', num2str(N_total_settings), '), Demo #', num2str(j), '/', num2str(N_demo_this_setting)]);

            [ sub_Ct_unroll_demo_cell{j,1}, ...
              global_traj_unroll_setting_cell{i,1}(1:3,j) ] = unrollObsAvoidViconTraj( data_global_coord.baseline(:,1), ...
                                                                                       data_global_coord.obs_avoid{setting_num,2}(:,j), ...
                                                                                       data_global_coord.obs_avoid{setting_num,1}, ...
                                                                                       data_global_coord.dt, ...
                                                                                       cart_coord_dmp_baseline_params, ...
                                                                                       loa_feat_methods, ...
                                                                                       loa_feat_param, ...
                                                                                       learning_param, ...
                                                                                       unrolling_param );
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
                if (exist([figure_path, '/trained_settings/'], 'dir') ~= 7) % if directory NOT exist
                    mkdir(figure_path, 'trained_settings'); % create directory
                end
                visualizeSetting( setting_num, 1, [figure_path,'/trained_settings/',figure_name], 0, ...
                                  unrolling_param.cart_coord_dmp_baseline_unroll_global_traj, ...
                                  global_traj_unroll_setting_cell{i,1}(1:3,1));
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

        [ mse_unroll, nmse_unroll ] = computeNMSE( Ct_unroll_aggregated, Ct_target );

        Ct_unroll_3D                = Ct_unroll;
        Ct_target_3D               	= reshape(Ct_target,3,size(Ct_target,1)/3).';
        [ mse_unroll_3D, nmse_unroll_3D ]   = computeNMSE( Ct_unroll_3D, Ct_target_3D );
    else
        [ mse_unroll, nmse_unroll ] = computeNMSE( Ct_unroll, Ct_target );
    end

    disp(['mse_unroll           = ', num2str(mse_unroll)]);
    disp(['nmse_unroll          = ', num2str(nmse_unroll)]);

    if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
        disp(['mse_unroll_per_dim   = ', num2str(mse_unroll_3D)]);
        disp(['nmse_unroll_per_dim  = ', num2str(nmse_unroll_3D)]);
    end

    % end of Unrolling on Trained Obstacle Settings
    
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
        normed_closest_distance_to_obs_per_unseen_setting     = zeros(N_unseen_settings, 1);
        normed_closest_distance_to_obs_per_unseen_setting_idx = zeros(N_unseen_settings, 1);
        final_distance_to_goal_vector               = zeros(N_unseen_settings, 1);
        point_obstacles_cart_position_global_cell   = cell(N_unseen_settings, 1);
        for i=1:N_unseen_settings
            sphere_params_global.center = sphere_obs_center_coords(:,i);
            tic
            disp(['   Unseen Setting #', num2str(i), '/', num2str(N_unseen_settings), ' ...']);
            [ sub_Ct_unroll_unseen_setting_cell{i,1}, ...
              global_traj_unroll_unseen_setting_cell{i,1}, ...
              local_traj_unroll_unseen_setting_cell{i,1}, ...
              normed_closest_distance_to_obs_traj_per_unseen_setting_cell{i,1}, ...
              final_distance_to_goal_vector(i,1), ...
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
                end
            end
        end
        [normed_closest_distance_to_obs_over_all_unseen_settings, normed_closest_distance_to_obs_over_all_unseen_settings_idx] = min(normed_closest_distance_to_obs_per_unseen_setting);
        percentage_hitting_obstacle     = (length(find(normed_closest_distance_to_obs_per_unseen_setting <= 0))/length(normed_closest_distance_to_obs_per_unseen_setting)) * 100.0;
        [worst_final_distance_to_goal, worst_final_distance_to_goal_unseen_setting_idx] = max(final_distance_to_goal_vector);
    end
    
    % end of Unrolling on Unseen Obstacle Settings
    
    %% Function Output Argument Settings
    
    performance_metric.nmse_learning        = nmse_learning;
    performance_metric.mse_learning         = mse_learning;
    performance_metric.nmse_unroll          = nmse_unroll;
    performance_metric.mse_unroll           = mse_unroll;
    if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
        performance_metric.nmse_learning_3D = nmse_learning_3D;
        performance_metric.mse_learning_3D  = mse_learning_3D;
        performance_metric.nmse_unroll_3D   = nmse_unroll_3D;
        performance_metric.mse_unroll_3D    = mse_unroll_3D;
    end
    if (is_unrolling_on_unseen_settings)
        performance_metric.normed_closest_distance_to_obs_over_all_unseen_settings      = normed_closest_distance_to_obs_over_all_unseen_settings;
        performance_metric.normed_closest_distance_to_obs_over_all_unseen_settings_idx  = normed_closest_distance_to_obs_over_all_unseen_settings_idx;
        performance_metric.normed_closest_distance_to_obs_per_unseen_setting            = normed_closest_distance_to_obs_per_unseen_setting;
        performance_metric.normed_closest_distance_to_obs_per_unseen_setting_idx        = normed_closest_distance_to_obs_per_unseen_setting_idx;
        performance_metric.percentage_hitting_obstacle                                  = percentage_hitting_obstacle;
        performance_metric.worst_final_distance_to_goal                     = worst_final_distance_to_goal;
        performance_metric.worst_final_distance_to_goal_unseen_setting_idx  = worst_final_distance_to_goal_unseen_setting_idx;
    end
    
    learning_unrolling_variables.X          = X;
    learning_unrolling_variables.Ct_target  = Ct_target;
    learning_unrolling_variables.Ct_fit     = Ct_fit;
    learning_unrolling_variables.Ct_unroll  = Ct_unroll;
    
    varargout(1)        = {performance_metric};
    if (nargout > 1)
        varargout(2)	= {learning_unrolling_variables};
    end
    
    % end of Function Output Argument Settings
end