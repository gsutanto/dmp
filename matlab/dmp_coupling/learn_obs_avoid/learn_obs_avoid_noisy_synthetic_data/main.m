close                       all;
clear                       all;
clc;

addpath('../data/');
addpath('../utilities/');

D                           = 3;    % Number of dimensions

n_rfs                       = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order                     = 1;    % DMP is using 2nd order canonical system

is_synthetic_weights_random = 0;

is_adding_random_variability= 1;

% feature method
loa_feat_method             = 0;
% loa_feat_method == 0: Akshara's Humanoids'14 features
% loa_feat_method == 1: Potential Field 2nd Dynamic Obst Avoid features

is_loading_precomputed_DMP_w= 0;
is_loading_precomputed_X_Ct = 0;

is_performing_ridge_reg     = 0;
is_performing_lasso_reg     = 0;

Wn                          = 0.02;

is_plot_processed_data      = 0; % processed = completeTrajectory() + diffnc + filtfilt
is_plot_primitives          = 0;
is_plot_synth_obs_avoid_traj= 1;
is_plot_ridge_reg_weights   = 0;
is_plot_reproduced_traj     = 1;

is_using_all_settings                   = 1;
is_using_all_obs_avoid_demos_per_setting= 1;

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
AF_H14_beta_phi1_phi2_low       = 5.0/pi;
AF_H14_beta_phi1_phi2_high      = 25.0/pi;
N_AF_H14_k_phi1_phi2_grid       = 5;
AF_H14_k_phi1_phi2_low          = 1.0;
AF_H14_k_phi1_phi2_high         = 41.0;

% Potential Field 2nd Model of Dynamic Obstacle Avoidance (by gsutanto) features:
N_PF_DYN2_beta_grid = 10;
PF_DYN2_beta_low    = 0.1;
PF_DYN2_beta_high   = 10.0;
N_PF_DYN2_k_grid    = 5;
PF_DYN2_k_low       = 0.1;
PF_DYN2_k_high      = 5.0;

point_feat_mode     = '_OBS_POINTS_AS_SEPARATE_FEATURES_';

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
                                                             point_feat_mode );

[ loa_feat_param ] = initializePF_DYN2LearnObsAvoidFeatParam( loa_feat_param, ...
                                                              N_PF_DYN2_beta_grid, ...
                                                              PF_DYN2_beta_low, ...
                                                              PF_DYN2_beta_high, ...
                                                              N_PF_DYN2_k_grid, ...
                                                              PF_DYN2_k_low, ...
                                                              PF_DYN2_k_high, ...
                                                              '_LINEAR_', ...
                                                              point_feat_mode );

% end of Obstacle Avoidance Features Grid Setting

%% Load the Data

% Use the following if filtering is needed:
[b,a]                   = butter(2, Wn);

data_temp   = load('data_multi_demo_static_preprocessed_02_2.mat');
data        = data_temp.data;

if (is_using_all_settings)
    end_idx_setting     = size(data,1);
else
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
end

% end of Load the Data

%% Learning Baseline Primitive

if (is_loading_precomputed_DMP_w == 0)
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
    
    save('wi.mat', 'wi', 'global_buffer');
else
    load('wi.mat');
end

% end of Learning Baseline Primitive

%% Creating Synthetic Obstacle Avoidance Trajectory from Known Model

N_loa_feat_vect_per_point   = getLOA_FeatureDimensionPerPoint(loa_feat_method, D, loa_feat_param);

% Synthetic Weights:
if (loa_feat_method == 0) % Akshara's Humanoids'14 features
    if (is_synthetic_weights_random)
        w_SYNTH         = 100*randn(2*N_loa_feat_vect_per_point,D);
    else
        w_SYNTH         = zeros(2*N_loa_feat_vect_per_point,D);

        w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+1,1)   = 100;
        w_SYNTH(N_loa_feat_vect_per_point + ...
                ((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+1,1)   = 200;
        w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+2,2)   = 300;
        w_SYNTH(N_loa_feat_vect_per_point + ...
                ((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+2,2)   = 400;
        w_SYNTH(((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+3,3)   = 100;
        w_SYNTH(N_loa_feat_vect_per_point + ...
                ((round((size(loa_feat_param.AF_H14_beta_phi1_phi2_vector,1)+1)/2)-1)*3)+3,3)   = 200;
            
        for i=1:size(w_SYNTH,1)
            for j=1:size(w_SYNTH,2)
                if (randn > 0.0)
                    w_SYNTH(i,j)    = w_SYNTH(i,j) + 50;
                end
            end
        end
    end
elseif (loa_feat_method == 1) % Potential Field 2nd Dynamic Obst Avoid features
    w_SYNTH     = 100*randn(2*N_loa_feat_vect_per_point,D);
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

%     if ((is_adding_random_variability) && (i ~= 1))
    if (is_adding_random_variability)
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
    [ X_gTruth_cell{1,i}, global_buffer{i,3}{1,1}, global_buffer{i,3}{2,1}, global_buffer{i,3}{3,1}, gT_ox3_cell{1,i}, gT_v3_cell{1,i}, Ct_SYNTH_cell{1,i} ] = constructNoisySynthObsAvoidSphereTraj( wi{1,i}, w_SYNTH, global_buffer{i,9}{1,1}, global_buffer{i,10}{1,1}, global_buffer{i,2}', traj_length_i, mean_dt{1,i}, c_order, global_buffer{i,4}, loa_feat_method, loa_feat_param );
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

% end of Creating Synthetic Obstacle Avoidance Trajectory from Known Model

%% Constructing Observed Obstacle Avoidance Features and Computing Target Coupling Term

if (is_loading_precomputed_X_Ct == 0)
    X_observed_cell         = cell(1,size(global_buffer,1));
    T_ox3_cell              = cell(1,size(global_buffer,1));
    T_v3_cell               = cell(1,size(global_buffer,1));
    Ct_target_cell          = cell(1,size(global_buffer,1));

    X_observed_stacked      = [];
    T_ox3_stacked           = [];
    T_v3_stacked            = [];
    Ct_target_stacked       = [];

    disp(['Constructing Observed Obstacle Avoidance Features and Computing Target Coupling Term']);

    for i=1:size(global_buffer,1)
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
    
    save('Ct.mat','X_observed_cell','Ct_target_cell',...
         'X_observed_stacked','Ct_target_stacked','Ct_target');
else
    load('Ct.mat');
end

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

%% Regression (with ARD)

% Performing ARD -- 1 setting
tic
disp(['Performing ARD (1 setting):']);
[ w_T_ard_single, nmse_learning_ard_single, Ct_T_fit_ard_single ] = learnUsingARD( X_observed, Ct_target );
toc

% Performing ARD -- all settings
tic
disp(['Performing ARD (all settings):']);
[ w_T_ard_all, nmse_learning_ard_all, Ct_T_fit_ard_all ] = learnUsingARD( X_observed_stacked, Ct_target_stacked );
toc

figure;
for d=1:D
    title_string    = ['w (with ARD) d=',num2str(d)];
    subplot(D,1,d);
    title(title_string);
    hold on;
    plot(w_T_ard_single(:,d));
    plot(w_T_ard_all(:,d));
    plot(w_SYNTH(:,d));
    legend('ARD single','ARD all','synthetic');
    hold off;
end

% end of Regression (with ARD)

%% Regression (with LASSO)

if (is_performing_lasso_reg)
    tic
    disp(['Performing LASSO (Observed):']);
    [ w_T_lasso, nmse_learning_lasso, Ct_T_fit_lasso ] = learnUsingLASSO( X_observed_stacked, Ct_target_stacked );
    toc

    figure;
    
    for d=1:D
        title_string    = ['w (with LASSO) d=',num2str(d)];
        subplot(D,1,d);
        hold on;
            plot(w_T_lasso(:,d));
            plot(w_SYNTH(:,d));
            legend('LASSO', 'synthetic');
            title(title_string);
        hold off;
    end
end

% end of Regression (with LASSO)

%% Comparison between Observed and Ground Truth, among Regression Methods

rank_XTX_single             = rank(X_observed.'*X_observed);
dim_XTX_single              = size(X_observed,2);
percentage_rank_XTX_single  = (100.0 * rank_XTX_single)/dim_XTX_single;
fprintf('rank(X^T*X)_single  = %d of matrix dimension %d (%f %%)\n', ...
        rank_XTX_single, dim_XTX_single, percentage_rank_XTX_single);
    
rank_XTX            = rank(X_observed_stacked.'*X_observed_stacked);
dim_XTX             = size(X_observed_stacked,2);
percentage_rank_XTX = (100.0 * rank_XTX)/dim_XTX;
fprintf('rank(X^T*X)         = %d of matrix dimension %d (%f %%)\n', ...
        rank_XTX, dim_XTX, percentage_rank_XTX);

mse_ox3     = mean(mean((gT_ox3_stacked-T_ox3_stacked).^2));
mse_v3      = mean(mean((gT_v3_stacked-T_v3_stacked).^2));
mse_X       = mean(mean((X_gTruth_stacked-X_observed_stacked).^2));
mse_Ct      = mean(mean((Ct_SYNTH_stacked-Ct_target_stacked).^2));
mse_w_single= mean(mean((w_SYNTH-w_T_ard_single).^2));
mse_w_all   = mean(mean((w_SYNTH-w_T_ard_all).^2));

disp(['mse(gT_ox3_stacked-T_ox3_stacked)        = ', num2str(mse_ox3)]);
disp(['mse(gT_v3_stacked-T_v3_stacked)          = ', num2str(mse_v3)]);
disp(['mse(X_gTruth_stacked-X_observed_stacked) = ', num2str(mse_X)]);
disp(['mse(Ct_SYNTH_stacked-Ct_target_stacked)  = ', num2str(mse_Ct)]);
disp(['mse(w_SYNTH-w_T_ard_all)                 = ', num2str(mse_w_all)]);

disp(['----------------']);
if (is_performing_ridge_reg)
    disp(['Ridge Regression']);
    disp(['nmse = ', num2str(nmse_learning_ridge)]);
    disp(['----------------']);
end
disp(['ARD (1 setting)']);
if (is_plot_reproduced_traj)
    % Unrolling (Observed, with ARD, 1 setting):
    [Y_u_ard_single,Yd_u_ard_single,Ydd_u_ard_single,Ct_u_ard_single,X_u_ard_single,U_ox3_ard_single,U_v3_ard_single,nmse_unrolling_ard_single] = unrollAndPlotObsAvoidSphereTraj(wi, w_T_ard_single, global_buffer, c_order, loa_feat_method, loa_feat_param, T_ox3_cell, T_v3_cell, X_observed_cell, Ct_target_cell, Ct_target_stacked, 'ARD 1 setting');
end
disp(['nmse learning                = ', num2str(nmse_learning_ard_single)]);
disp(['nmse unrolling               = ', num2str(nmse_unrolling_ard_single)]);
disp(['max(w_T_ard_single)          = ', num2str(max(max(w_T_ard_single)))]);
disp(['min(w_T_ard_single)          = ', num2str(min(min(w_T_ard_single)))]);
disp(['norm(w_T_ard_single)         = ', num2str(norm(w_T_ard_single))]);
disp(['mse(w_SYNTH-w_T_ard_single)  = ', num2str(mse_w_single)]);
disp(['----------------']);
disp(['ARD (all settings)']);
if (is_plot_reproduced_traj)
    % Unrolling (Observed, with ARD, all settings):
    [Y_u_ard_all,Yd_u_ard_all,Ydd_u_ard_all,Ct_u_ard_all,X_u_ard_all,U_ox3_ard_all,U_v3_ard_all,nmse_unrolling_ard_all] = unrollAndPlotObsAvoidSphereTraj(wi, w_T_ard_all, global_buffer, c_order, loa_feat_method, loa_feat_param, T_ox3_cell, T_v3_cell, X_observed_cell, Ct_target_cell, Ct_target_stacked, 'ARD all settings');
end
disp(['nmse learning                = ', num2str(nmse_learning_ard_all)]);
disp(['nmse unrolling               = ', num2str(nmse_unrolling_ard_all)]);
disp(['max(w_T_ard_all)             = ', num2str(max(max(w_T_ard_all)))]);
disp(['min(w_T_ard_all)             = ', num2str(min(min(w_T_ard_all)))]);
disp(['norm(w_T_ard_all)            = ', num2str(norm(w_T_ard_all))]);
disp(['mse(w_SYNTH-w_T_ard_all)     = ', num2str(mse_w_all)]);
disp(['----------------']);
if (is_performing_lasso_reg)
    disp(['LASSO']);
    disp(['nmse = ', num2str(nmse_learning_lasso)]);
    disp(['----------------']);
end

idx_limit_plot  = size(Ct_target,1);

figure;
for d=1:D
    subplot(D,1,d);
    title_string    = ['Ct Fit d=',num2str(d)];
    hold on;
    plot(Ct_target(1:idx_limit_plot,d));
    plot(Ct_T_fit_ard_all(1:idx_limit_plot,d));
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
    plot(Ct_T_fit_ard_all(1:idx_limit_plot,d));
    plot(Ct_u_ard_all{1,1}{1,1}(:,d));
    legend('Target', 'Fit ARD', 'Unrolling');
    hold off;
    title(title_string);
end