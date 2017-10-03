close                       all;
clear                       all;
clc;

addpath('../../../../utilities/');
addpath('../../data/');
addpath('../../utilities/');

D                           = 3;    % Number of dimensions

n_rfs                       = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order                     = 1;    % DMP is using 2nd order canonical system

data_temp                   = load('data_multi_demo_static_preprocessed_02_1.mat');
data                        = data_temp.data;

% feature method
loa_feat_method             = 0;
% loa_feat_method == 0: Akshara's Humanoids'14 features
% loa_feat_method == 1: Potential Field 2nd Dynamic Obst Avoid features

obs_point_feat_mode         = '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_';
% obs_point_feat_mode == '_OBS_POINTS_AS_SEPARATE_FEATURES_':
%   Two points: one at the sphere obstacle center and 
%               one at the sphere obstacle surface closest to the
%               end-effector are considered for separate feature computation
% obs_point_feat_mode == '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_':
%   Two points: one at the sphere obstacle center and 
%               one at the sphere obstacle surface closest to the
%               end-effector are considered for feature computation, 
%               and their contributions are summed in the end
% obs_point_feat_mode == '_USE_ONLY_CLOSEST_SURFACE_OBS_POINT_':
%   One point: a point at the sphere obstacle surface 
%              closest to the end-effector are considered for feature computation

is_performing_lasso_reg     = 1;

Wn                          = 0.02;

is_plot_processed_data      = 0; % processed = completeTrajectory() + diffnc + filtfilt
is_plot_primitives          = 0;

is_using_all_settings                   = 0;
is_using_all_obs_avoid_demos_per_setting= 1;

% for single setting/obstacle position:
setting_selection_number    = 1;
% setting_selection_number    = 6;

k_fold_array                = [5,4,2];

flag                        = 'master_arm';
object                      = 'sph'; % 'ell' , 'cyl'

if (strcmp(flag, 'master_arm') == 1)
    % MasterArm robot's sampling rate is 420.0 Hz
    dt      = 1.0/420.0;
else
    if(strcmp(object,'sph'))
        % sphere data is sampled at 1000 Hz
        dt  = 0.001;
    else
        % ellipsoid and cylinder sampled at 100Hz
        dt  = 0.01;
    end
end

freq                        = 1/dt;

%% Obstacle Avoidance Features Grid Setting

% Akshara-Franzi (Humanoids'14) features:
N_AF_H14_beta_phi1_phi2_grid 	= 10;
% AF_H14_beta_phi1_phi2_low       = 7.0/pi;  % best so far
% AF_H14_beta_phi1_phi2_high      = 20.0/pi; % best so far
AF_H14_beta_phi1_phi2_low       = 7.0/pi;
AF_H14_beta_phi1_phi2_high      = 20.0/pi;
N_AF_H14_k_phi1_phi2_grid       = 5;
% AF_H14_k_phi1_phi2_low          = 0.1; % best so far
% AF_H14_k_phi1_phi2_high         = 5.0; % best so far
AF_H14_k_phi1_phi2_low          = 0.1;
AF_H14_k_phi1_phi2_high         = 5.0;

% Potential Field 2nd Model of Dynamic Obstacle Avoidance (by gsutanto) features:
N_PF_DYN2_beta_grid = 10;
PF_DYN2_beta_low    = 0.1;
PF_DYN2_beta_high   = 10.0;
N_PF_DYN2_k_grid    = 5;
PF_DYN2_k_low       = 0.1;
PF_DYN2_k_high      = 5.0;

loa_feat_param     = initializeLearnObsAvoidFeatParam();
[ loa_feat_param ] = initializeAF_H14LearnObsAvoidFeatParam( loa_feat_param, ...
                                                             N_AF_H14_beta_phi1_phi2_grid, ...
                                                             AF_H14_beta_phi1_phi2_low, ...
                                                             AF_H14_beta_phi1_phi2_high, ...
                                                             N_AF_H14_k_phi1_phi2_grid, ...
                                                             AF_H14_k_phi1_phi2_low, ...
                                                             AF_H14_k_phi1_phi2_high );

[ loa_feat_param ] = initializePF_DYN2LearnObsAvoidFeatParam( loa_feat_param, ...
                                                              N_PF_DYN2_beta_grid, ...
                                                              PF_DYN2_beta_low, ...
                                                              PF_DYN2_beta_high, ...
                                                              N_PF_DYN2_k_grid, ...
                                                              PF_DYN2_k_low, ...
                                                              PF_DYN2_k_high );

% end of Obstacle Avoidance Features Grid Setting

%% Load the Data

% Use the following if filtering is needed:
[b,a]                   = butter(2, Wn);

if (is_using_all_settings)
    end_idx_setting     = size(data,1);
else
    data_temp           = data;
    clear               data;
    data                = cell(1,size(data_temp,2));
    for i=1:size(data,2)
        data{1,i}       = data_temp{setting_selection_number,i};
    end

    end_idx_setting     = 1;
end
global_buffer   = cell(end_idx_setting,10);
num_OA_demo     = zeros(end_idx_setting,1);

% Data Pre-processing:
for i=1:end_idx_setting
    if (is_using_all_obs_avoid_demos_per_setting)
        num_OA_demo(i)  = size(data{i,3},2);
    else
        num_OA_demo(i)  = 1;
    end
    
    % Baseline (without Obstacle) Behavior
    global_buffer{i,5}  = zeros(1,size(data{i,1},2));   % Baseline taus
    global_buffer{i,6}  = zeros(1,size(data{i,1},2));   % Baseline dts
    for j=1:size(data{i,1},2)
        % Baseline (without Obstacle) Demonstrations
        global_buffer{i,1}{1,j} = data{i,1}{1,j};
        global_buffer{i,1}{2,j} = data{i,1}{2,j};
        global_buffer{i,1}{3,j} = data{i,1}{3,j};
        
        % Baseline (without Obstacle) taus and dts
        global_buffer{i,5}(1,j) = (size(global_buffer{i,1}{1,j},1)-1)*dt;
        global_buffer{i,6}(1,j) = dt;
    end
    
    % Obstacle Description
    global_buffer{i,2}  = data{i,2};    % Obstacle Center Coordinate
    global_buffer{i,4}  = data{i,4};    % Obstacle Radius

    % Obstacle Avoidance Behavior
    global_buffer{i,7}  = zeros(1,num_OA_demo(i));   % Obstacle Avoidance taus
    global_buffer{i,8}  = zeros(1,num_OA_demo(i));   % Obstacle Avoidance dts
    global_buffer{i,9}  = cell(1,num_OA_demo(i));    % Obstacle Avoidance start
    global_buffer{i,10} = cell(1,num_OA_demo(i));    % Obstacle Avoidance goal
    for j=1:num_OA_demo(i)
        % Obstacle Avoidance Demonstrations
        global_buffer{i,3}{1,j} = data{i,3}{1,j};
        global_buffer{i,3}{2,j} = data{i,3}{2,j};
        global_buffer{i,3}{3,j} = data{i,3}{3,j};

        % Obstacle Avoidance taus and dts
        global_buffer{i,7}(1,j) = (size(global_buffer{i,3}{1,j},1)-1)*dt;
        global_buffer{i,8}(1,j) = dt;

        % Start and Goal Position of the Demonstrated Obstacle
        % Avoidance Behavior
        global_buffer{i,9}{1,j} = global_buffer{i,3}{1,j}(1,:)'; 
        global_buffer{i,10}{1,j}= global_buffer{i,3}{1,j}(end,:)';
    end

    if (is_plot_processed_data)
        if (i==1)
            for d=1:D
                figure;
                for ord=1:3 % y, yd, ydd
                    subplot(3,1,ord)
                    if (ord==1)
                        title_string = ['d=',num2str(d)];
                        title(title_string);
                    end
                    hold on;
                    plot(data{i,3}{ord,1}(:,d));
                    plot(global_buffer{i,3}{ord,1}(:,d));
                    y_string = ['order=',num2str(ord)];
                    ylabel(y_string);
                    legend('raw','filtered');
                    hold off;
                end
            end
            keyboard;
        end
    end
end

% end of Load the Data

%% Learning Baseline Primitive

wi                  = cell(1,size(global_buffer,1));
Yifit               = cell(1,size(global_buffer,1));
Ydifit              = cell(1,size(global_buffer,1));
Yddifit             = cell(1,size(global_buffer,1));
Fifit               = cell(1,size(global_buffer,1));
mean_start          = cell(1,size(global_buffer,1));
mean_goal           = cell(1,size(global_buffer,1));
mean_dt             = cell(1,size(global_buffer,1));
mean_traj_length    = cell(1,size(global_buffer,1));

disp(['Learning Baseline Primitive']);

for i=1:end_idx_setting
    disp(['    ', num2str(i)]);
    [ wi{1,i}, Yifit{1,i}, Ydifit{1,i}, Yddifit{1,i}, Fifit{1,i}, mean_start{1,i}, mean_goal{1,i}, mean_dt{1,i} ] = learnPrimitiveMulti( global_buffer{i,1}(1,:), global_buffer{i,1}(2,:), global_buffer{i,1}(3,:), global_buffer{i,5}, global_buffer{i,6}, n_rfs, c_order );
    mean_traj_length{1,i}   = size(Fifit{1,i},1);
end

if (is_plot_primitives)
    figure;
    subplot(2,2,1);
        hold on;
            axis equal;
            plot3(Yifit{1,1}(:,1),Yifit{1,1}(:,2),Yifit{1,1}(:,3),'r');
            for j=1:size(global_buffer{1,1},2)
                plot3(global_buffer{1,1}{1,j}(:,1),global_buffer{1,1}{1,j}(:,2),global_buffer{1,1}{1,j}(:,3),'b');
            end
            title('Yi: data vs fit');
            legend('fit');
        hold off;
    subplot(2,2,2);
        hold on;
            axis equal;
            plot3(Ydifit{1,1}(:,1),Ydifit{1,1}(:,2),Ydifit{1,1}(:,3),'r');
            for j=1:size(global_buffer{1,1},2)
                plot3(global_buffer{1,1}{2,j}(:,1),global_buffer{1,1}{2,j}(:,2),global_buffer{1,1}{2,j}(:,3),'b');
            end
            title('Ydi: data vs fit');
            legend('fit');
        hold off;
    subplot(2,2,3);
        hold on;
            axis equal;
            plot3(Yddifit{1,1}(:,1),Yddifit{1,1}(:,2),Yddifit{1,1}(:,3),'r');
            for j=1:size(global_buffer{1,1},2)
                plot3(global_buffer{1,1}{3,j}(:,1),global_buffer{1,1}{3,j}(:,2),global_buffer{1,1}{3,j}(:,3),'b');
            end
            title('Yddi: data vs fit');
            legend('fit');
        hold off;
    subplot(2,2,4);
        hold on;
            axis equal;
            plot3(Fifit{1,1}(:,1),Fifit{1,1}(:,2),Fifit{1,1}(:,3),'r');
            title('Fi fit');
        hold off;
end

% end of Learning Baseline Primitive

%% Constructing Observed Obstacle Avoidance Features and Computing Target Coupling Term

X_observed_cell         = cell(1,size(global_buffer,1));
T_ox3_cell              = cell(1,size(global_buffer,1));
T_v3_cell               = cell(1,size(global_buffer,1));
Ct_target_cell          = cell(1,size(global_buffer,1));

X_observed_stacked      = [];
T_ox3_stacked           = [];
T_v3_stacked            = [];
Ct_target_stacked       = [];

disp(['Constructing Observed Obstacle Avoidance Features and Computing Target Coupling Term']);

for i=1:end_idx_setting
    disp(['    ', num2str(i)]);
    X_observed_cell{1,i}        = cell(1,size(global_buffer{i,3},2));
    Ct_target_cell{1,i}         = cell(1,size(global_buffer{i,3},2));

    for j=1:size(global_buffer{i,3},2)
        [ X_observed_cell{1,i}{1,j}, T_ox3_cell{1,i}{1,j}, T_v3_cell{1,i}{1,j} ]= constructObsAvoidSphereFeatMat3D( global_buffer{i,3}{1,j}, global_buffer{i,3}{2,j}, global_buffer{i,2}', global_buffer{i,8}(1,j), global_buffer{i,4}, loa_feat_method, loa_feat_param );
        [ Ct_target_cell{1,i}{1,j} ] = computeDMPCtTarget( global_buffer{i,3}{1,j}, global_buffer{i,3}{2,j}, global_buffer{i,3}{3,j}, wi{1,i}, n_rfs, global_buffer{i,9}{1,j}, global_buffer{i,10}{1,j}, global_buffer{i,8}(1,j), c_order );

        T_ox3_stacked               = [T_ox3_stacked; T_ox3_cell{1,i}{1,j}];
        T_v3_stacked                = [T_v3_stacked; T_v3_cell{1,i}{1,j}];
        X_observed_stacked          = [X_observed_stacked; X_observed_cell{1,i}{1,j}];
        Ct_target_stacked           = [Ct_target_stacked; Ct_target_cell{1,i}{1,j}];
    end
end

X_observed          = X_observed_cell{1,1}{1,1};
T_ox3               = T_ox3_cell{1,1}{1,1};
T_v3                = T_v3_cell{1,1}{1,1};
Ct_target           = Ct_target_cell{1,1}{1,1};

[ linespec_codes ]  = generateLinespecCodes();

i = 1;
n_interp        = length(Ct_target_cell{1,1}{1,1}(:,1));
count_figure    = 0;
h = figure;
for d=1:D
    subplot(D,1,d);
    hold on;
        for j=1:size(global_buffer{i,3},2)
            stretched_traj  = stretchTrajectory( Ct_target_cell{1,i}{1,j}(:,d), n_interp );
            plot_handle{j}  = plot(stretched_traj,linespec_codes{1,j});
            plot_legend{j}  = num2str(j);
        end
        legend([plot_handle{:}], plot_legend{:});
    hold off;
    drawnow;
end
saveas(h,['Figure',num2str(count_figure),'.jpg']);
count_figure    = count_figure + 1;

% end of Constructing Observed Obstacle Avoidance Features and Computing Target Coupling Term

%% Regression

learning_methods_cell   = {'ARD'};
if (is_performing_lasso_reg)
    learning_methods_cell{1,size(learning_methods_cell,2)+1}    = 'LASSO';
end

tic
[ nmse_train_cell, nmse_test_cell, w_cell, w_per_dim_cell, max_w_cell, min_w_cell, sparsity_w_cell ] = learnAndTestGeneralizationKFoldCrossValidation( X_observed_stacked, Ct_target_stacked, learning_methods_cell, k_fold_array );
toc

% for i=1:size(global_buffer{1,3},2)
%     holdout_idx = i;
%     type=1;
%     figure;
%     for d=1:D
%         subplot(D,1,d);
%         if (d==1)
%             title(['run', num2str(i)]);
%         end
%         hold on
%         for j=1:size(global_buffer{1,3},2)
%             plot(global_buffer{1,3}{type,j}(:,d),'b');
%         end
%         plot(global_buffer{1,3}{type,holdout_idx}(:,d),'g');
%         hold off
%     end
% end

nmse_train_ave_cell = cell(size(learning_methods_cell));
nmse_test_ave_cell  = cell(size(learning_methods_cell));
max_w_ave_cell      = cell(size(learning_methods_cell));
min_w_ave_cell      = cell(size(learning_methods_cell));
sparsity_w_ave_cell = cell(size(learning_methods_cell));

for l_idx=1:size(learning_methods_cell,2)
    nmse_train_ave_cell{1,l_idx}    = zeros(size(k_fold_array,2),D);
    nmse_test_ave_cell{1,l_idx}     = zeros(size(k_fold_array,2),D);
    max_w_ave_cell{1,l_idx}         = zeros(size(k_fold_array,2),D);
    min_w_ave_cell{1,l_idx}         = zeros(size(k_fold_array,2),D);
    sparsity_w_ave_cell{1,l_idx}    = zeros(size(k_fold_array,2),D);
    for f_idx=1:size(k_fold_array,2)
        nmse_train_ave_cell{1,l_idx}(f_idx,:)   = mean(nmse_train_cell{f_idx,l_idx},1);
        nmse_test_ave_cell{1,l_idx}(f_idx,:)    = mean(nmse_test_cell{f_idx,l_idx},1);
        max_w_ave_cell{1,l_idx}(f_idx,:)        = mean(max_w_cell{f_idx,l_idx},1);
        min_w_ave_cell{1,l_idx}(f_idx,:)        = mean(min_w_cell{f_idx,l_idx},1);
        sparsity_w_ave_cell{1,l_idx}(f_idx,:)   = mean(sparsity_w_cell{f_idx,l_idx},1);
    end

    h = figure;
    for d=1:D
        subplot(D,1,d);
        if (d==1)
            title([learning_methods_cell{1,l_idx}, ': Average k-fold CV NMSE Comparison between Training vs Test']);
        end
        hold on;
            bar_aug     = [nmse_train_ave_cell{1,l_idx}(:,d),nmse_test_ave_cell{1,l_idx}(:,d)];
            bar(k_fold_array, bar_aug);
            legend('train','test');
            if (d==3)
                xlabel(['k-fold']);
            end
            ylabel(['dim ',num2str(d)]);
        hold off;
    end
    saveas(h,[learning_methods_cell{1,l_idx},'_Figure',num2str(count_figure),'.jpg']);
    count_figure    = count_figure + 1;

    h = figure;
    for d=1:D
        subplot(D,1,d);
        if (d==1)
            title([learning_methods_cell{1,l_idx}, ': Average k-fold CV Training NMSE']);
        end
        hold on;
            bar(k_fold_array, nmse_train_ave_cell{1,l_idx}(:,d));
            if (d==3)
                xlabel(['k-fold']);
            end
            ylabel(['dim ',num2str(d)]);
        hold off;
    end
    saveas(h,[learning_methods_cell{1,l_idx}, '_Figure',num2str(count_figure),'.jpg']);
    count_figure    = count_figure + 1;

    for f_idx=1:size(k_fold_array,2)
        h=figure;
        for d=1:D
            subplot(D,1,d);
            if (d==1)
                title([learning_methods_cell{1,l_idx}, ': Weights Variability Across Dimensions for ',num2str(k_fold_array(1,f_idx)),'-Fold Cross Validation']);
            end
            hold on;
                errorbar(1:size(w_per_dim_cell{f_idx,l_idx}{1,d},1), ...
                         mean(w_per_dim_cell{f_idx,l_idx}{1,d},2), ...
                         std(w_per_dim_cell{f_idx,l_idx}{1,d},[],2), '.');
                ylabel(['dim ',num2str(d)]);
            hold off;
        end
        saveas(h,[learning_methods_cell{1,l_idx}, '_Figure',num2str(count_figure),'.jpg']);
        count_figure    = count_figure + 1;
    end

    h=figure;
    for d=1:D
        subplot(D,1,d);
        if (d==1)
            title([learning_methods_cell{1,l_idx}, ': (Average) Max Weights Variability Across Dimensions']);
        end
        hold on;
            bar(k_fold_array, max_w_ave_cell{1,l_idx}(:,d));
            if (d==3)
                xlabel(['k-fold']);
            end
            ylabel(['dim ',num2str(d)]);
        hold off;
    end
    saveas(h,[learning_methods_cell{1,l_idx}, '_Figure',num2str(count_figure),'.jpg']);
    count_figure    = count_figure + 1;

    h=figure;
    for d=1:D
        subplot(D,1,d);
        if (d==1)
            title([learning_methods_cell{1,l_idx}, ': (Average) Min Weights Variability Across Dimensions']);
        end
        hold on;
            bar(k_fold_array, min_w_ave_cell{1,l_idx}(:,d));
            if (d==3)
                xlabel(['k-fold']);
            end
            ylabel(['dim ',num2str(d)]);
        hold off;
    end
    saveas(h,[learning_methods_cell{1,l_idx}, '_Figure',num2str(count_figure),'.jpg']);
    count_figure    = count_figure + 1;

    h=figure;
    for d=1:D
        subplot(D,1,d);
        if (d==1)
            title([learning_methods_cell{1,l_idx}, ': (Average) Sparsity of Weights Variability Across Dimensions']);
        end
        hold on;
            bar(k_fold_array, sparsity_w_ave_cell{1,l_idx}(:,d));
            if (d==3)
                xlabel(['k-fold']);
            end
            ylabel(['dim ',num2str(d)]);
        hold off;
    end
    saveas(h,[learning_methods_cell{1,l_idx}, '_Figure',num2str(count_figure),'.jpg']);
    count_figure    = count_figure + 1;
end

h = figure;
for d=1:D
    subplot(D,1,d);
    if (d==1)
        title(['All Learning Methods: Average k-fold CV NMSE Comparison between Training vs Test']);
    end
    hold on;
        bar_aug     = [];
        legend_cell = cell(1,2*size(learning_methods_cell,2));
        for l_idx=1:size(learning_methods_cell,2)
            bar_aug = [bar_aug, nmse_train_ave_cell{1,l_idx}(:,d), nmse_test_ave_cell{1,l_idx}(:,d)];
            legend_cell{1,(l_idx-1)*2+1}    = [learning_methods_cell{1,l_idx}, ' train'];
            legend_cell{1,(l_idx-1)*2+2}    = [learning_methods_cell{1,l_idx}, ' test'];
        end
        bar(k_fold_array, bar_aug);
        legend(legend_cell{:});
        if (d==3)
            xlabel(['k-fold']);
        end
        ylabel(['dim ',num2str(d)]);
    hold off;
end
saveas(h,['All_Figure',num2str(count_figure),'.jpg']);
count_figure    = count_figure + 1;

keyboard;

% end of Regression
