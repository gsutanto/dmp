% clear  	all;
close   all;
clc;

rng(1234)

addpath('../utilities/');

[~, name] = system('hostname');
if (strcmp(name(1,1:end-1), 'gsutanto-ThinkPad-T430s') == 1)
    figure_path    = '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/';
else
    figure_path    = 'figures_single/';
end

performance_metric_evaluation_mode                                  = 1;
unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;
unrolling_param.is_comparing_with_cpp_implementation                = 0;

unrolling_param.is_plot_unrolling  	= 1;
loa_feat_methods    = 7;
feat_constraint_mode                = '_CONSTRAINED_';
learning_param.learning_constraint_mode = '_NONE_';
learning_param.N_iter_ard = 200;

unrolling_param.verify_NN_inference    	= 0;

D               = 3;

n_rfs           = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order         = 1;    % DMP is using 2nd order canonical system

loa_feat_methods_to_be_evaluated    = [5, 7];
N_feat_methods  = size(loa_feat_methods_to_be_evaluated, 2);

%% Data Creation/Loading

data_filepath                       = '../data/data_multi_demo_vicon_static_global_coord.mat';

load(data_filepath);

%% Baseline Primitive Learning
disp('Processing Local Coordinate System for Demonstrated Baseline Trajectories ...');
[ cart_coord_dmp_baseline_params, ...
  unrolling_param.cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(data_global_coord.baseline, data_global_coord.dt, n_rfs, c_order);

% end of Baseline Primitive Learning

%% Test settings
% test_settings  = [8, 10, 26, 27, 28,29,30,31,44,45,46,47,83,84,98,99,100,102,116,117,118,119,120,122,136,137,157,172,173,174,175,176,177,178,190,193,194,195,197,209,210]; 
% Sphere
sets = [1 2 5 6 7 9 11 12 13 14 15 16 17 18 19 20 23 24 26 27 28 29 30 31 33 35 37 39 41 42 43 45 46 47 50 51 53 54 56 57 58 59 62 63 65 68];
% Cube and cylinder
sets = [sets 72 75 77,79,80,83,87,88,90,92,94,97,98,100,101,102,103,104,105,107,110, 112, 117, 119,120,121,123,128, 134,137,138,143,152,153,154,156,160,164,172,174,175,176,177,179,182,189,193,194,195,196,197,200,212,213];
test_settings = sets;


N_settings = length(test_settings);

% end of initializing settings
%% Running training
for idx=1:N_settings
    i = test_settings(idx);
    disp(['>>Evaluating Performance Metric for Setting #', num2str(i), '/', num2str(N_settings), ' ...']);
    selected_obs_avoid_setting_numbers  = [i];
    max_num_trajs_per_setting           = 100;

    for j=1:N_feat_methods
        disp(['>>>>>Feature Method #', num2str(loa_feat_methods_to_be_evaluated(1, j)), ' (', num2str(j), '/', num2str(N_feat_methods), ') ...']);
        [ performance_metric, ...
          learning_unrolling_variables, ...
          learning_param ]  = learnAndUnrollObsAvoidViconDataSetting( data_global_coord, ...
                                                                      cart_coord_dmp_baseline_params, ...
                                                                      loa_feat_methods_to_be_evaluated(1, j), ...
                                                                      feat_constraint_mode, ...
                                                                      learning_param, ...
                                                                      unrolling_param, ...
                                                                      selected_obs_avoid_setting_numbers, ...
                                                                      max_num_trajs_per_setting, ...
                                                                      figure_path, 0 );
        icra_performance_metric_each_single_setting{i, j}   = performance_metric;
    end

    if ((mod(i,3) == 0) || (i == N_settings))  % save data everytime 3 settings of evaluations are completed
        save('icra_performance_metric_each_single_setting.mat', 'icra_performance_metric_each_single_setting');

        if ((mod(i,15) == 0) || (i == N_settings))
            copyfile('icra_performance_metric_each_single_setting.mat', ...
                     figure_path);
        end
    end
end
save('icra_performance_metric_each_single_setting.mat', 'icra_performance_metric_each_single_setting');


%% Plot results generated

methods = {'humanoids', 'NN'};
close all
load('icra_performance_metric_each_single_setting.mat')
    
for j = 1:N_feat_methods
    for idx = 1:N_settings
        i = test_settings(idx);
        data = icra_performance_metric_each_single_setting{i,j};
        nmse_learn(:,idx,j) =  data.nmse_learning;
        nmse_unroll(:,idx,j) = data.nmse_unroll;
        min_dist(idx,j) = data.trained_settings.normed_closest_distance_to_obs_overall_train_demos_per_setting;
        human_dist(idx,j) = data.human_demo.normed_closest_dist_to_obs_human_1st_demo_over_all_settings;
        min_baseline(idx,j) = data.baseline.normed_closest_distance_to_obs_human_1st_demo_setting;
        dist_to_goal(idx,j) = data.trained_settings.worst_final_distance_to_goal_per_trained_setting;
    end

figure, 
method = methods{j};
subplot(311), hold on, title([method ' nmse learning dim 1'])
plot(nmse_learn(1,:), '.-')
subplot(312), hold on, title([method ' nmse learning dim 2'])
plot(nmse_learn(2,:), '.-')
subplot(313), hold on, title([method ' nmse learning dim 3'])
plot(nmse_learn(3,:), '.-')

% figure, 
% subplot(311), hold on, title([method ' nmse unroll dim 1'])
% plot(nmse_unroll(1,:), '.-')
% subplot(312), hold on, title([method ' nmse unroll dim 2'])
% plot(nmse_unroll(2,:), '.-')
% subplot(313), hold on, title([method ' nmse unroll dim 3'])
% plot(nmse_unroll(3,:), '.-')

X = [0 0.2 0.5 0.7 1.0 2.0 inf];
figure, 
subplot(311), hold on, title([method ' histogram nmse learn dim1 '])
histogram(nmse_learn(1,:),X)
subplot(312), hold on, title([method ' histogram nmse learn dim2 '])
histogram(nmse_learn(2,:),X)
subplot(313), hold on, title([method ' histogram nmse learn dim3 '])
histogram(nmse_learn(3,:),X)

% figure, 
% subplot(311), hold on, title('neural network histogram nmse unroll dim 1')
% histogram(nmse_unroll(1,:),X)
% subplot(312), hold on, title('neural network histogram nmse unroll dim 2')
% histogram(nmse_unroll(2,:),X)
% subplot(313), hold on, title('neural network histogram nmse unroll dim 3')
% histogram(nmse_unroll(3,:),X)

figure
subplot(211), hold on, title([method ' minimum distance to obstacle'])
plot(min_dist(:,j), '.-')
plot(min_baseline(:,j), '.-r')
plot(human_dist(:,j), '.-k')
legend('unrolled', 'baseline', 'human demo')
subplot(212), hold on, title([method ' final distance from goal'])
plot(dist_to_goal(:,j), '.-')
end
%% Generate csv file to create tikz plots

for j=1:N_feat_methods
    method = methods{j};
    x = 1:length(min_dist);
    y1 = min_dist(:,j);
    y2 = min_baseline(:,j);
    y3 = human_dist(:,j);

    csvwrite([method '_dist_to_obs.csv'], [x' y1 y2 y3]);
    
    y4 = nmse_learn(1,:,j);
    y5 = nmse_learn(2,:,j);
    y6 = nmse_learn(3,:,j);
    csvwrite([method '_nmse_train.csv'], [y4' y5' y6']);
end
