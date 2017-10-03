% clear  	all;
close   all;
clc;

rng(1234)

addpath('../utilities/');

[~, name] = system('hostname');
if (strcmp(name(1,1:end-1), 'gsutanto-ThinkPad-T430s') == 1)
    figure_path    = '/home/gsutanto/Desktop/CLMC/Publications/Humanoids16/data/';
else
    figure_path    = 'figures_multi/';
end

performance_metric_evaluation_mode                                  = 2;
unrolling_param.is_unrolling_only_1st_demo_each_trained_settings    = 1;
unrolling_param.is_comparing_with_cpp_implementation                = 0;

unrolling_param.is_plot_unrolling  	= 1;
loa_feat_methods    = 7;
feat_constraint_mode                = '_CONSTRAINED_';
learning_param.learning_constraint_mode = '_NONE_';

unrolling_param.verify_NN_inference    	= 0;

D               = 3;

n_rfs           = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order         = 1;    % DMP is using 2nd order canonical system

loa_feat_methods_to_be_evaluated    = [7];
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
settings_selection_mode = 1;
% test_settings  = [8, 10, 26, 27, 28,29,30,31,44,45,46,47,83,84,98,99,100,102,116,117,118,119,120,122,136,137,157,172,173,174,175,176,177,178,190,193,194,195,197,209,210]; 
% sets = [];
% Sphere
% sets = [1 2 5 6 7 9 11 12 13 14 15 16 17 18 19 20 23 24 26 27 28 29 30 31 33 35 37 39 41 42 43 45 46 47 50 51 53 54 56 57 58 59 62 63 65 68];
% Cube 
% sets = [sets 72 75 77,79,80,83,87,88,90,92,94,97,98,100,101,102,103,104,105,107,110, 112, 117, 119,120,121,123,128, 134,137,138,143];
%and cylinder
% sets = [sets 152,153,154,156,160,164,172,174,175,176,177,179,182,189,193,194,195,196,197,200,212,213];
sets = [149,150,152,153,154,156,158,160,161,163,164,166, 168, 170,171, 172,175,176,177,178,179,181,182,185, 187, 188, 189,190, 193,194,195,196,197,199,200,205,208, 209,212,213];
% gsutanto's cylinder set:
% sets = setdiff([sets, 149:222], [149, 155]);
% sets = [190:197];
% sets = [174, 175, 176, 193, 194, 210, 211,  150,151,152,159,160,161]; % best gsutanto
% sets = [193, 194];
% sets = [194];
% sets = [176];
test_settings = sets;

N_settings = length(test_settings);

% end of initializing settings
%% Running training
selected_obs_avoid_setting_numbers = test_settings;
max_num_trajs_per_setting           = 10; % 1;
is_unrolling_on_unseen_settings = 1;

for j=1:N_feat_methods
    disp(['>>>Feature Method #', num2str(loa_feat_methods_to_be_evaluated(1, j)), ' (', num2str(j), '/', num2str(N_feat_methods), ') ...']);
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
                                                                  figure_path, is_unrolling_on_unseen_settings );
    icra_performance_metric_multi_settings{1, j}   = performance_metric;
end

save('icra_performance_metric_multi_settings_sph.mat', 'icra_performance_metric_multi_settings');

copyfile('icra_performance_metric_multi_settings_sph.mat', figure_path);


%% Plot results generated for trained settings
load('icra_performance_metric_multi_settings_sph.mat')

for j = 1:N_feat_methods
    data = icra_performance_metric_multi_settings{1,j};
    nmse_learn =  data.nmse_learning;
    nmse_unroll = data.nmse_unroll;
    min_dist = data.trained_settings.normed_closest_distance_to_obs_overall_train_demos_per_setting;
    human_dist = data.human_demo.normed_closest_distance_to_obs_human_1st_demo_setting;
    min_baseline = data.baseline.normed_closest_distance_to_obs_human_1st_demo_setting;
    dist_to_goal = data.trained_settings.worst_final_distance_to_goal_per_trained_setting;
end

disp(['nmse_learning          = ', num2str(nmse_learn)]);
disp(['nmse_unroll          = ', num2str(nmse_unroll)]);

figure
subplot(211), hold on, title('trained - minimum distance to obstacle')
plot(min_dist, '.-')
plot(min_baseline, '.-r')
plot(human_dist, '.-k')
legend('unrolled', 'baseline', 'human demo')
subplot(212), hold on, title('trained - final distance from goal')
plot(dist_to_goal, '.-')

disp(['mean min distance to obstacle          = ', num2str(mean(min_dist))]);
disp(['mean distance to goal          = ', num2str(mean(dist_to_goal))]);

%% Plot results generated for unseen settings
load('icra_performance_metric_multi_settings_sph.mat')


for j = 1:N_feat_methods
    data = icra_performance_metric_multi_settings{1,j};
    N_unseen_settings = length(data.unseen_settings.normed_closest_distance_to_obs_per_unseen_setting);
    min_dist = data.unseen_settings.normed_closest_distance_to_obs_per_unseen_setting;
    dist_to_goal = data.unseen_settings.final_distance_to_goal_vector_per_unseen_setting;
end

disp(['nmse_learning          = ', num2str(nmse_learn)]);
disp(['nmse_unroll          = ', num2str(nmse_unroll)]);

figure
subplot(211), hold on, title('unseen - minimum distance to obstacle')
plot(min_dist, '.-')
subplot(212), hold on, title('unseen - final distance from goal')
plot(dist_to_goal, '.-')

disp(['mean min distance to obstacle          = ', num2str(mean(min_dist))]);
disp(['mean distance to goal          = ', num2str(mean(dist_to_goal))]);


