close           all;
clear           all;

c_order         = 1;

is_plot_fit_traj            = 0;
is_plot_forcing_term_fit    = 0;
is_plot_synth_obs_avoid_traj= 0;
is_plot_gT_vs_obs_first     = 1;
is_plot_gT_vs_obs_stacked   = 0;

%% Generate Obstacles
n_rfs           = 25;
nxgrid          = 5;
nygrid          = 5;

[obs_xgrid, obs_ygrid] = meshgrid(linspace(0.0,2.0,nxgrid),linspace(0.0,2.0,nygrid));
obs_grid        = [reshape(obs_xgrid, 1, nxgrid*nygrid);...
    reshape(obs_ygrid, 1, nxgrid*nygrid)];

obs_list        = [[2.05,1.5]',obs_grid];

n_beta_grid     = 3;
n_k_grid        = 3;
[beta_mgrid, k_mgrid] = meshgrid(linspace(6.0/pi,14.0/pi,n_beta_grid),linspace(10,30,n_k_grid));
beta_grid       = reshape(beta_mgrid, n_beta_grid*n_k_grid, 1);
k_grid          = reshape(k_mgrid, n_beta_grid*n_k_grid, 1);

w_ct_synth         = zeros(2*n_beta_grid*n_k_grid,2);

w_ct_synth(n_beta_grid*n_k_grid,1)   = 1500;
w_ct_synth(n_beta_grid*n_k_grid,2) = 500;

% end of Generate Obstacles

%% Load the Data
dt              = 1.0/420.0;
load('traj.mat');
sample_traj_2D  = traj(:,1:2);

sample_traj_2D_d    = zeros(size(sample_traj_2D));
sample_traj_2D_dd   = zeros(size(sample_traj_2D));
for d=1:2
    sample_traj_2D_d(:,d)   = diffnc(sample_traj_2D(:,d),dt);
    sample_traj_2D_dd(:,d)  = diffnc(sample_traj_2D_d(:,d),dt);
end

traj_length         = size(sample_traj_2D,1);
% end of Load the Data

%% Learning Baseline
[ w_primitive, Yi] = learnPrimitive( sample_traj_2D, sample_traj_2D_d, sample_traj_2D_dd, ...
    n_rfs, sample_traj_2D(1,:), sample_traj_2D(end,:), traj_length, dt, c_order );
start_pos = Yi(1,:);
goal_pos = Yi(end,:);
clear sample_traj_2D sample_traj_2D_d sample_traj_2D_dd

%% Creating Synthetic Obstacle Avoidance Trajectory from Known Model
X_gTruth_cell   = cell(size(obs_list,2),1);
Yo_cell         = cell(size(obs_list,2),1);
Ydo_cell        = cell(size(obs_list,2),1);
Yddo_cell       = cell(size(obs_list,2),1);
Ct_SYNTH_cell   = cell(size(obs_list,2),1);

for i=1:size(obs_list,2)
    [ X_gTruth_cell{i,1}, Yo_cell{i,1}, Ydo_cell{i,1}, Yddo_cell{i,1}, ~, ~, Ct_SYNTH_cell{i,1} ] = ...
        constructSynthObsAvoidPointTraj( w_primitive, w_ct_synth, start_pos, goal_pos,...
        obs_list(:,i), traj_length, dt, beta_grid, k_grid, c_order );
end
X_gTruth_stacked    = cell2mat(X_gTruth_cell);
Ct_SYNTH_stacked    = cell2mat(Ct_SYNTH_cell);
 
% end of Creating Synthetic Obstacle Avoidance Trajectory from Known Model

%% Constructing Observed Obstacle Avoidance Features
X_observed_cell     = cell(size(obs_list,2),1);
Ct_target_cell      = cell(size(obs_list,2),1);

for i=1:size(obs_list,2)
    [ X_observed_cell{i,1}] = constructObsAvoidPointFeatMat2D( Yo_cell{i,1}, Ydo_cell{i,1}, ...
        obs_list(:,i), beta_grid, k_grid );
    [ Ct_target_cell{i,1}] = computeDMPCtTarget( Yo_cell{i,1}, Ydo_cell{i,1}, ...
        Yddo_cell{i,1}, w_primitive, n_rfs, start_pos, goal_pos, traj_length, dt, c_order );
end
X_observed_stacked = cell2mat(X_observed_cell);
Ct_target_stacked = cell2mat(Ct_target_cell);
% end of Constructing Observed Obstacle Avoidance Features

%% Regression (Ridge)
tic
disp(['Performing Ridge Regression (X_gTruth vs Ct_SYNTH):']);
[ w_gT_SYNTH, nmseS_gT, Ct_gT_fit ] = learnUsingRidgeRegression( X_gTruth_stacked, Ct_SYNTH_stacked );
toc

tic
disp(['Performing Ridge Regression (X_observed vs Ct_target):']);
[ w_T, nmse_T, Ct_T_fit ] = learnUsingRidgeRegression( X_observed_stacked, Ct_target_stacked );
toc

% end of Regression (Ridge)

%% Regression (with ARD)

tic
disp(['Performing ARD (Ground Truth):']);
[ w_gT_SYNTH_ard, nmseS_gT_ard, Ct_gT_fit_ard ] = learnUsingARD( X_gTruth_stacked, Ct_SYNTH_stacked );
toc

tic
disp(['Performing ARD (Observed):']);
[ w_T_ard, nmse_T_ard, Ct_T_fit_ard ] = learnUsingARD( X_observed_stacked, Ct_target_stacked );
toc

% end of Regression (with ARD)

%% Regression (with LASSO)

tic
disp(['Performing LASSO (Ground Truth):']);
[ w_gT_SYNTH_lasso, nmseS_gT_lasso, Ct_gT_fit_lasso ] = learnUsingLASSO( X_gTruth_stacked, Ct_SYNTH_stacked );
toc

tic
disp(['Performing LASSO (Observed):']);
[ w_T_lasso, nmse_T_lasso, Ct_T_fit_lasso ] = learnUsingLASSO( X_observed_stacked, Ct_target_stacked );
toc

% end of Regression (with LASSO)

%% Comparison between Observed and Ground Truth, among Regression Methods

fprintf('rank(XX_gT) = %d\n', rank(X_gTruth_stacked.'*X_gTruth_stacked));
fprintf('rank(XX_T)  = %d\n', rank(X_observed_stacked.'*X_observed_stacked));

disp(['----------------']);
disp(['Ridge Regression']);
disp(['nmse (Ground Truth) = ', num2str(nmseS_gT)]);
disp(['nmse (Observed)     = ', num2str(nmse_T)]);
disp(['----------------']);
disp(['ARD']);
disp(['nmse (Ground Truth) = ', num2str(nmseS_gT_ard)]);
disp(['nmse (Observed)     = ', num2str(nmse_T_ard)]);
disp(['----------------']);
disp(['LASSO']);
disp(['nmse (Ground Truth) = ', num2str(nmseS_gT_lasso)]);
disp(['nmse (Observed)     = ', num2str(nmse_T_lasso)]);
disp(['----------------']);

figure,
plot(w_ct_synth(:,1),'g+'), hold on, plot(w_T_ard(:,1),'x'), plot(w_T_lasso(:,1), 'co'), hold off
figure,
plot(w_ct_synth(:,2),'g+'), hold on, plot(w_T_ard(:,2),'x'), plot(w_T_lasso(:,2), 'co'), hold off
% plot(Ct_SYNTH_cell{1,1}(:,1),'g'), hold on, plot(Ct_target_cell{1,1}(:,1),'.')