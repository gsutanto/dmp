close                       all;
clear                       all;
clc;

addpath('../data/');
addpath('../utilities/');

D                           = 3;    % Number of dimensions

n_rfs                       = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order                     = 1;    % DMP is using 2nd order canonical system

is_operating_on_real_data   = 1;
is_synthetic_weights_random = 0;

% data_file   = {'data_multi_demo_static_preprocessed_02_1.mat','data_sph_new.mat'};
data_file   = {'data_multi_demo_static_preprocessed_02_2.mat','data_sph_new.mat'};

is_adding_random_variability= 1;

is_using_gsutanto_data      = 1;
is_using_akshara_data       = 0;

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

is_performing_ridge_reg     = 0;
is_performing_bayesian_reg  = 1;
is_performing_lasso_reg     = 0;

is_loading_lasso_weights    = 0;

Wn                          = 0.02;

is_plot_processed_data      = 0; % processed = completeTrajectory() + diffnc + filtfilt
is_plot_primitives          = 0;
is_plot_synth_obs_avoid_traj= 1;
is_plot_ridge_reg_weights   = 0;
is_plot_reproduced_traj     = 1;

is_using_all_settings                   = 1;
is_using_all_obs_avoid_demos_per_setting= 1;

% for single setting/obstacle position:
setting_selection_number    = 1;
% setting_selection_number    = 6;

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

if (is_operating_on_real_data == 0)
    % Akshara-Franzi (Humanoids'14) features:
    N_AF_H14_beta_phi1_phi2_grid    = 3;
    AF_H14_beta_phi1_phi2_low       = 5.0/pi;
    AF_H14_beta_phi1_phi2_high      = 25.0/pi;
    N_AF_H14_k_phi1_phi2_grid       = 3;
    AF_H14_k_phi1_phi2_low          = 1.0;
    AF_H14_k_phi1_phi2_high         = 41.0;
    
    % Potential Field 2nd Model of Dynamic Obstacle Avoidance (by gsutanto) features:
    N_PF_DYN2_beta_grid = 10;
    PF_DYN2_beta_low    = 0.1;
    PF_DYN2_beta_high   = 10.0;
    N_PF_DYN2_k_grid    = 5;
    PF_DYN2_k_low       = 0.1;
    PF_DYN2_k_high      = 5.0;
else
    % Akshara-Franzi (Humanoids'14) features:
    N_AF_H14_beta_phi1_phi2_grid	= 10;
%     AF_H14_beta_phi1_phi2_low       = 7.0/pi;  % best so far
%     AF_H14_beta_phi1_phi2_high      = 20.0/pi; % best so far
    AF_H14_beta_phi1_phi2_low       = 7.0/pi;
    AF_H14_beta_phi1_phi2_high      = 20.0/pi;
    N_AF_H14_k_phi1_phi2_grid       = 5;
%     AF_H14_k_phi1_phi2_low          = 0.1; % best so far
%     AF_H14_k_phi1_phi2_high         = 5.0; % best so far
    AF_H14_k_phi1_phi2_low          = 0.1;
    AF_H14_k_phi1_phi2_high         = 5.0;
%     N_AF_H14_beta_phi1_phi2_grid    = 13;
%     AF_H14_beta_phi1_phi2_low       = 3.0/pi;
%     AF_H14_beta_phi1_phi2_high      = 30.0/pi;
%     N_AF_H14_k_phi1_phi2_grid       = 8;
%     AF_H14_k_phi1_phi2_low          = 0.01;
%     AF_H14_k_phi1_phi2_high         = 5.0;
    
    % Potential Field 2nd Model of Dynamic Obstacle Avoidance (by gsutanto) features:
    N_PF_DYN2_beta_grid = 10;
    PF_DYN2_beta_low    = 0.1;
    PF_DYN2_beta_high   = 10.0;
    N_PF_DYN2_k_grid    = 5;
    PF_DYN2_k_low       = 0.1;
    PF_DYN2_k_high      = 5.0;
end

loa_feat_param     = initializeLearnObsAvoidFeatParam();
[ loa_feat_param ] = initializeAF_H14LearnObsAvoidFeatParam( loa_feat_param, ...
                                                             N_AF_H14_beta_phi1_phi2_grid, ...
                                                             AF_H14_beta_phi1_phi2_low, ...
                                                             AF_H14_beta_phi1_phi2_high, ...
                                                             N_AF_H14_k_phi1_phi2_grid, ...
                                                             AF_H14_k_phi1_phi2_low, ...
                                                             AF_H14_k_phi1_phi2_high, ...
                                                             0, 0, 0, ...
                                                             '_LINEAR_', ...
                                                             obs_point_feat_mode );

[ loa_feat_param ] = initializePF_DYN2LearnObsAvoidFeatParam( loa_feat_param, ...
                                                              N_PF_DYN2_beta_grid, ...
                                                              PF_DYN2_beta_low, ...
                                                              PF_DYN2_beta_high, ...
                                                              N_PF_DYN2_k_grid, ...
                                                              PF_DYN2_k_low, ...
                                                              PF_DYN2_k_high, ...
                                                              '_LINEAR_', ...
                                                              obs_point_feat_mode );

% end of Obstacle Avoidance Features Grid Setting

%% Load the Data

% Use the following if filtering is needed:
[b,a]                   = butter(2, Wn);

data                    = cell(0,0);
for i=1:size(data_file,2)
    if (((is_using_gsutanto_data) && (i==1)) || ...
        ((is_using_akshara_data) && (i==2)))
        data_temp       = load(data_file{1,i});
        data            = [data; data_temp.data];
    end
end

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
    
    if (is_operating_on_real_data)
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

%% Creating Synthetic Obstacle Avoidance Trajectory from Known Model

if (is_operating_on_real_data == 0) % operating on synthetic data
    
    % number of obstacle points considered
    if (strcmp(loa_feat_param.point_feat_mode, '_OBS_POINTS_AS_SEPARATE_FEATURES_') == 1)
        nP                      = 2;
    else % for '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_' or '_USE_ONLY_CLOSEST_SURFACE_OBS_POINT_'
        nP                      = 1;
    end
    N_loa_feat_vect_per_point   = getLOA_FeatureDimensionPerPoint(loa_feat_method, D, loa_feat_param);
    
    % Synthetic Weights:
    if (loa_feat_method == 0) % Akshara's Humanoids'14 features
        if (is_synthetic_weights_random)
            w_SYNTH         = 100*randn(nP*N_loa_feat_vect_per_point,D);
        else
            w_SYNTH         = zeros(nP*N_loa_feat_vect_per_point,D);
            
            if (nP == 2)
                w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+1,1)   = 100;
                w_SYNTH(N_loa_feat_vect_per_point + ...
                        ((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+1,1)   = 200;
                w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+2,2)   = 300;
                w_SYNTH(N_loa_feat_vect_per_point + ...
                        ((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+2,2)   = 400;
                w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+3,3)   = 100;
                w_SYNTH(N_loa_feat_vect_per_point + ...
                        ((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+3,3)   = 200;
            elseif (nP == 1)
                w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+1,1)   = 200;
                w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+2,2)   = 400;
                w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+3,3)   = 200;
            end
        end
    elseif (loa_feat_method == 1) % Potential Field 2nd Dynamic Obst Avoid features
        w_SYNTH     = 100*randn(nP*N_loa_feat_vect_per_point,D);
    end
    
    X_gTruth_cell   = cell(1,size(global_buffer,1));
    gT_ox3_cell     = cell(1,size(global_buffer,1));
    gT_v3_cell      = cell(1,size(global_buffer,1));
    Ct_SYNTH_cell   = cell(1,size(global_buffer,1));

    X_gTruth_stacked    = [];
    gT_ox3_stacked      = [];
    gT_v3_stacked       = [];
    Ct_SYNTH_stacked    = [];

    for i=1:end_idx_setting
        
        global_buffer{i,2}      = data{i,2};                        % Obstacle Center Coordinate
        
        if ((is_adding_random_variability) && (i ~= 1))
            global_buffer{i,4}  = (1.0+abs(0.5*rand))*data{i,4};    % Obstacle Radius
            
            % Start and Goal Position of the Demonstrated Obstacle
            % Avoidance Behavior
            global_buffer{i,9}{1,1}     = mean_start{1,i} + (0.1*rand(size(mean_start{1,i}))); 
            global_buffer{i,10}{1,1}    = mean_goal{1,i} + (0.1*rand(size(mean_goal{1,i})));
            
            % Vary the tau (movement duration) by multiplying traj_length with a random number:
            traj_length_i               = round((1.0+abs(rand))*mean_traj_length{1,i});
        else
            global_buffer{i,4}  = data{i,4};   % Obstacle Radius
            
            % Start and Goal Position of the Demonstrated Obstacle
            % Avoidance Behavior
            global_buffer{i,9}{1,1}     = mean_start{1,i}; 
            global_buffer{i,10}{1,1}    = mean_goal{1,i};
            
            traj_length_i               = mean_traj_length{1,i};
        end
        
        % Obstacle Avoidance Behavior
        [ X_gTruth_cell{1,i}, global_buffer{i,3}{1,1}, global_buffer{i,3}{2,1}, global_buffer{i,3}{3,1}, gT_ox3_cell{1,i}, gT_v3_cell{1,i}, Ct_SYNTH_cell{1,i} ] = constructSynthObsAvoidSphereTraj( wi{1,i}, w_SYNTH, global_buffer{i,9}{1,1}, global_buffer{i,10}{1,1}, global_buffer{i,2}', traj_length_i, mean_dt{1,i}, c_order, global_buffer{i,4}, loa_feat_method, loa_feat_param );
        X_gTruth_stacked    = [X_gTruth_stacked; X_gTruth_cell{1,i}];
        gT_ox3_stacked      = [gT_ox3_stacked; gT_ox3_cell{1,i}];
        gT_v3_stacked       = [gT_v3_stacked; gT_v3_cell{1,i}];
        Ct_SYNTH_stacked    = [Ct_SYNTH_stacked; Ct_SYNTH_cell{1,i}];
        
        % Obstacle Avoidance tau and dt
        global_buffer{i,7}(1,1) = (size(global_buffer{i,3}{1,1},1)-1)*dt;
        global_buffer{i,8}(1,1) = dt;
    end

    X_gTruth        = X_gTruth_cell{1,1};
    Yo              = global_buffer{1,3}{1,1};
    Ydo             = global_buffer{1,3}{2,1};
    Yddo            = global_buffer{1,3}{3,1};
    gT_ox3          = gT_ox3_cell{1,1};
    gT_v3           = gT_v3_cell{1,1};
    Ct_SYNTH        = Ct_SYNTH_cell{1,1};

    if (is_plot_synth_obs_avoid_traj)
        figure;
        hold on;
        axis equal;
        plot3(Yifit{1,1}(:,1),Yifit{1,1}(:,2),Yifit{1,1}(:,3),'b');
        plot3(Yo(:,1),Yo(:,2),Yo(:,3),'g');
        plot_sphere(global_buffer{1,4}, global_buffer{1,2}(1,1), global_buffer{1,2}(1,2), global_buffer{1,2}(1,3));
        xlabel('x');
        ylabel('y');
        zlabel('z');
        legend('baseline traj','obst avoid traj','obstacle');
        title('baseline vs obst avoid traj');
        hold off;
        
%         keyboard;
    end

    % dlmwrite('synthetic_obst_avoid_traj.txt', Yo, 'delimiter', ' ');
end

% end of Creating Synthetic Obstacle Avoidance Trajectory from Known Model

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
        X_observed_stacked      = [X_observed_stacked; X_observed_cell{1,i}{1,j}];
        T_ox3_stacked           = [T_ox3_stacked; T_ox3_cell{1,i}{1,j}];
        T_v3_stacked            = [T_v3_stacked; T_v3_cell{1,i}{1,j}];
        Ct_target_stacked       = [Ct_target_stacked; Ct_target_cell{1,i}{1,j}];
    end
end

X_observed          = X_observed_cell{1,1}{1,1};
T_ox3               = T_ox3_cell{1,1}{1,1};
T_v3                = T_v3_cell{1,1}{1,1};
Ct_target           = Ct_target_cell{1,1}{1,1};

% end of Constructing Observed Obstacle Avoidance Features and Computing Target Coupling Term

%% Regression (Ridge)

if (is_performing_ridge_reg)
    tic
    disp(['Performing Ridge Regression (X_observed vs Ct_target):']);
    [ w_T_ridge, nmse_learning_ridge, Ct_T_fit_ridge ] = learnUsingRidgeRegression( X_observed_stacked, Ct_target_stacked );
    toc

    if (is_plot_ridge_reg_weights)
        figure;
        hold on;
        for d=1:D
            title_string    = ['w (with Ridge Reg) d=',num2str(d)];
            subplot(D,1,d); plot(w_T_ridge(:,d)); title(title_string);
        end
        hold off;
    end
end

% end of Regression (Ridge)

%% Regression (with Bayesian Regression)

tic
disp(['Performing Bayesian Regression (Observed):']);
[ w_T_br, nmse_learning_br, Ct_T_fit_br ] = learnUsingBayesianRegression( X_observed_stacked, Ct_target_stacked );
toc

figure;
for d=1:D
    subplot(D,1,d);
    title(['w (with Bayesian Regression) d=',num2str(d)]);
    hold on;
    plot(w_T_br(:,d));
    if (is_operating_on_real_data == 0)
        plot(w_SYNTH(:,d));
        legend('Bayesian Regression','synthetic');
    else
        legend('Bayesian Regression');
    end
    hold off;
end
drawnow;

% end of Regression (with Bayesian Regression)

%% Regression (with ARD)

tic
disp(['Performing ARD (Observed):']);
[ w_T_ard, nmse_learning_ard, Ct_T_fit_ard ] = learnUsingARD( X_observed_stacked, Ct_target_stacked );
toc

figure;
for d=1:D
    subplot(D,1,d);
    title(['w (with ARD) d=',num2str(d)]);
    hold on;
    plot(w_T_ard(:,d));
    if (is_operating_on_real_data == 0)
        plot(w_SYNTH(:,d));
        legend('ARD','synthetic');
    else
        legend('ARD');
    end
    hold off;
end
drawnow;

% end of Regression (with ARD)

%% Regression (with LASSO)

if (is_performing_lasso_reg) 
    if (is_loading_lasso_weights == 0)
        tic
        disp(['Performing LASSO (Observed):']);
        [ w_T_lasso, nmse_learning_lasso, Ct_T_fit_lasso ] = learnUsingLASSO( X_observed_stacked, Ct_target_stacked );
        toc
    else
        load('lasso_results_all_settings_data_02_2_obs_point_method_0.mat');
    end

    figure;
    hold on;
    for d=1:D
        subplot(D,1,d); plot(w_T_lasso(:,d)); title(['w (with LASSO) d=',num2str(d)]);
    end
    hold off;
    drawnow;
end

% end of Regression (with LASSO)

%% Comparison between Observed and Ground Truth, among Regression Methods

rank_XTX            = rank(X_observed_stacked.'*X_observed_stacked);
dim_XTX             = size(X_observed_stacked,2);
percentage_rank_XTX = (100.0 * rank_XTX)/dim_XTX;
fprintf('rank(X^T*X)  = %d of matrix dimension %d (%f %%)\n', ...
        rank_XTX, dim_XTX, percentage_rank_XTX);

if (is_operating_on_real_data == 0)
    mse_ox3 = mean(mean((gT_ox3_stacked-T_ox3_stacked).^2));
    mse_v3  = mean(mean((gT_v3_stacked-T_v3_stacked).^2));
    mse_X   = mean(mean((X_gTruth_stacked-X_observed_stacked).^2));
    mse_Ct  = mean(mean((Ct_SYNTH_stacked-Ct_target_stacked).^2));
    mse_w   = mean(mean((w_SYNTH-w_T_ard).^2));
    
    disp(['mse(gT_ox3_stacked-T_ox3_stacked)        = ', num2str(mse_ox3)]);
    disp(['mse(gT_v3_stacked-T_v3_stacked)          = ', num2str(mse_v3)]);
    disp(['mse(X_gTruth_stacked-X_observed_stacked) = ', num2str(mse_X)]);
    disp(['mse(Ct_SYNTH_stacked-Ct_target_stacked)  = ', num2str(mse_Ct)]);
    disp(['mse(w_SYNTH-w_T_ard)                     = ', num2str(mse_w)]);
end

disp(['----------------']);
if (is_performing_ridge_reg)
    disp(['Ridge Regression']);
    if (is_plot_reproduced_traj) % Unrolling (Observed, with Ridge Regression)
        [Y_u_ridge,Yd_u_ridge,Ydd_u_ridge,Ct_u_ridge,X_u_ridge,U_ox3_ridge,U_v3_ridge,nmse_unrolling_ridge] = unrollAndPlotObsAvoidSphereTraj(wi, w_T_ridge, global_buffer, c_order, loa_feat_method, loa_feat_param, T_ox3_cell, T_v3_cell, X_observed_cell, Ct_target_cell, Ct_target_stacked, 'Ridge Regression');
    end
    disp([' ']);
    disp(['nmse learning  = ', num2str(nmse_learning_ridge)]);
    disp(['nmse unrolling = ', num2str(nmse_unrolling_ridge)]);
    disp(['----------------']);
end
if (is_performing_bayesian_reg)
    disp(['Bayesian Regression']);
    if (is_plot_reproduced_traj) % Unrolling (Observed, with Bayesian Regression)
        [Y_u_br,Yd_u_br,Ydd_u_br,Ct_u_br,X_u_br,U_ox3_br,U_v3_br,nmse_unrolling_br] = unrollAndPlotObsAvoidSphereTraj(wi, w_T_br, global_buffer, c_order, loa_feat_method, loa_feat_param, T_ox3_cell, T_v3_cell, X_observed_cell, Ct_target_cell, Ct_target_stacked, 'Bayesian Regression');
    end
    disp([' ']);
    disp(['nmse learning  = ', num2str(nmse_learning_br)]);
    disp(['nmse unrolling = ', num2str(nmse_unrolling_br)]);
    disp(['----------------']);
end
disp(['ARD']);
if (is_plot_reproduced_traj) % Unrolling (Observed, with ARD)
    [Y_u_ard,Yd_u_ard,Ydd_u_ard,Ct_u_ard,X_u_ard,U_ox3_ard,U_v3_ard,nmse_unrolling_ard] = unrollAndPlotObsAvoidSphereTraj(wi, w_T_ard, global_buffer, c_order, loa_feat_method, loa_feat_param, T_ox3_cell, T_v3_cell, X_observed_cell, Ct_target_cell, Ct_target_stacked, 'ARD');
end
disp([' ']);
disp(['nmse learning  = ', num2str(nmse_learning_ard)]);
disp(['nmse unrolling = ', num2str(nmse_unrolling_ard)]);
disp(['----------------']);
if (is_performing_lasso_reg)
    disp(['LASSO']);
    if (is_plot_reproduced_traj) % Unrolling (Observed, with LASSO)
        [Y_u_lasso,Yd_u_lasso,Ydd_u_lasso,Ct_u_lasso,X_u_lasso,U_ox3_lasso,U_v3_lasso,nmse_unrolling_lasso] = unrollAndPlotObsAvoidSphereTraj(wi, w_T_lasso, global_buffer, c_order, loa_feat_method, loa_feat_param, T_ox3_cell, T_v3_cell, X_observed_cell, Ct_target_cell, Ct_target_stacked, 'LASSO');
    end
    disp([' ']);
    disp(['nmse learning  = ', num2str(nmse_learning_lasso)]);
    disp(['nmse unrolling = ', num2str(nmse_unrolling_lasso)]);
    disp(['----------------']);
end

idx_limit_plot  = size(Ct_target,1);

figure;
for d=1:D
    subplot(D,1,d);
    title_string    = ['Ct Fit d=',num2str(d)];
    hold on;
    plot(Ct_target(1:idx_limit_plot,d));
    plot(Ct_T_fit_ard(1:idx_limit_plot,d));
    if (is_performing_ridge_reg)
        plot(Ct_T_fit_ridge(1:idx_limit_plot,d));
        legend('Target', 'Fit Ridge', 'Fit ARD');
    else
        legend('Target', 'Fit ARD');
    end
    hold off;
    title(title_string);
end

% end of Comparison between Observed and Ground Truth, among Regression Methods

%% Plot All Ct's Together

figure;
for d=1:D
    subplot(D,1,d);
    title_string    = ['Ct Fit d=',num2str(d)];
    hold on;
    plot(Ct_target(1:idx_limit_plot,d));
    plot(Ct_T_fit_ard(1:idx_limit_plot,d));
    plot(Ct_u_ard{1,1}{1,1}(:,d));
    legend('Target', 'Fit ARD', 'Unrolling');
    hold off;
    title(title_string);
end