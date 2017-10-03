close           all;
clear           all;
clc;

addpath('../../utilities/');
addpath('../data/');

n_rfs           = 25;
c_order         = 1;

traj_no                     = 2;
is_min_jerk                 = 0;
is_plot_fit_traj            = 0;
is_plot_forcing_term_fit    = 0;
is_plot_synth_obs_avoid_traj= 0;
is_plot_gT_vs_obs_first     = 1;
is_plot_gT_vs_obs_stacked   = 0;

%% Generate Obstacles

obs_radius          = 0.035;

if (traj_no==0)
    nxgrid          = 5;
    nygrid          = 5;

    [obs_xgrid, obs_ygrid] = meshgrid(linspace(0.0,2.0,nxgrid),linspace(0.0,2.0,nygrid));
    obs_grid        = [reshape(obs_xgrid, 1, nxgrid*nygrid);...
                       reshape(obs_ygrid, 1, nxgrid*nygrid)];
                   
    obs_list        = [[0.5,0.3525]',obs_grid];
elseif (traj_no==1)
    nxgrid          = 20;
    nygrid          = 20;
    
    [obs_xgrid, obs_ygrid] = meshgrid(linspace(0.0,1.0,nxgrid),linspace(0.0,1.0,nygrid));
    obs_grid        = [reshape(obs_xgrid, 1, nxgrid*nygrid);...
                       reshape(obs_ygrid, 1, nxgrid*nygrid)];
                   
    % obs_list        = [[0.525,0.49]',obs_grid];
    obs_list        = [[0.525,0.49; 0.3,0.4; 0.75,0.4; 0.4, 0.45; 0.6, 0.475;...
                       0.8,0.35; 0.15,0.3; 0.225,0.35; 0.7,0.45; 0.85,0.3;...
                       0.45,0.475; 0.35,0.425; 0.475,0.45; 0.25,0.4]',obs_grid];
    % obs_list        = [0.525,0.49]';
elseif (traj_no==2)
    nxgrid          = 5;
    nygrid          = 5;

    [obs_xgrid, obs_ygrid] = meshgrid(linspace(-0.5,1.0,nxgrid),linspace(0.2,0.6,nygrid));
    obs_grid        = [reshape(obs_xgrid, 1, nxgrid*nygrid);...
                       reshape(obs_ygrid, 1, nxgrid*nygrid)];
                   
    obs_list        = [[2.0,1.5]',obs_grid];
end

n_beta_grid     = 3;
n_k_grid        = 3;
[beta_mgrid, k_mgrid] = meshgrid(linspace(6.0/pi,14.0/pi,n_beta_grid),linspace(10,30,n_k_grid));
beta_grid       = reshape(beta_mgrid, n_beta_grid*n_k_grid, 1);
k_grid          = reshape(k_mgrid, n_beta_grid*n_k_grid, 1);
% beta            = 10/pi;  % synthetic data parameter
% k               = 20;     % synthetic data parameter

w_SYNTH         = zeros(2*n_beta_grid*n_k_grid,2);

if (traj_no==0)
    w_SYNTH(1,1)                            = 5;
    w_SYNTH(round(size(w_SYNTH,1)/4),1)     = 23;
    w_SYNTH(n_beta_grid*n_k_grid,1)         = 38;
    w_SYNTH(n_beta_grid*n_k_grid+1,2)       = 88;
    w_SYNTH(round(size(w_SYNTH,1)*3/4),2)   = 40;
    w_SYNTH(end,2)                          = 19;
elseif (traj_no==1)
    w_SYNTH(n_beta_grid*n_k_grid,1)   = 70;
    w_SYNTH(n_beta_grid*n_k_grid+1,2) = 70;
elseif (traj_no==2)
    w_SYNTH(1,1)                            = 90;
    w_SYNTH(round(size(w_SYNTH,1)/4),1)     = 50;
    w_SYNTH(n_beta_grid*n_k_grid,1)         = 500;
    w_SYNTH(n_beta_grid*n_k_grid+1,2)       = 70;
    w_SYNTH(round(size(w_SYNTH,1)*3/4),2)   = 90;
    w_SYNTH(end,2)                          = 120;
end

% end of Generate Obstacles

%% Load the Data

if (traj_no==0)
    dt              = 0.01;
    sample_traj_2D  = dlmread('sample_traj_2D_03.txt');
elseif (traj_no==1)
    dt              = 0.01;
    sample_traj_2D  = dlmread('sample_traj_2D.txt');

    % filter the data:
    data = {};
    data{1} = sample_traj_2D;
    [b,a] = butter(2,0.1);
    for i = 1:size(data,2)
        tx = data{i};
        tf = tx';
        for j = 1:2
            x = tx(:,j);
            tf(j,:) = filtfilt(b,a,x');
        end
        data{i} = tf';
    end

    sample_traj_2D  = data{1};
elseif (traj_no==2)
    dt              = 1.0/420.0;
    load('traj.mat');
    sample_traj_2D  = traj(:,1:2);
end

sample_traj_2D_d    = zeros(size(sample_traj_2D));
sample_traj_2D_dd   = zeros(size(sample_traj_2D));
for d=1:2
    sample_traj_2D_d(:,d)   = diffnc(sample_traj_2D(:,d),dt);
    sample_traj_2D_dd(:,d)  = diffnc(sample_traj_2D_d(:,d),dt);
end

traj_length         = size(sample_traj_2D,1);

% end of Load the Data

%% Learning Baseline

[ wi, Yi, Ydi, Yddi, Fti, Fi ] = learnPrimitive( sample_traj_2D, sample_traj_2D_d, sample_traj_2D_dd, n_rfs, sample_traj_2D(1,:), sample_traj_2D(end,:), traj_length, dt, c_order );

if (is_plot_fit_traj)
    figure;
    subplot(2,2,1);
        hold on;
            plot(sample_traj_2D(:,1),sample_traj_2D(:,2),'r');
            plot(Yi(:,1),Yi(:,2),'b');
            scatter(obs_list(1,:),obs_list(2,:),'co');
            title('Yi: data vs fit');
            legend('data','fit','obstacle');
        hold off;
    subplot(2,2,2);
        hold on;
            plot(sample_traj_2D_d(:,1),sample_traj_2D_d(:,2),'r');
            plot(Ydi(:,1),Ydi(:,2),'b');
            title('Ydi: data vs fit');
            legend('data','fit');
        hold off;
    subplot(2,2,3);
        hold on;
            plot(sample_traj_2D_dd(:,1),sample_traj_2D_dd(:,2),'r');
            plot(Yddi(:,1),Yddi(:,2),'b');
            title('Yddi: data vs fit');
            legend('data','fit');
        hold off;
    subplot(2,2,4);
        hold on;
            plot(Fti(:,1),Fti(:,2),'r');
            plot(Fi(:,1),Fi(:,2),'b');
            title('Fi: target vs fit');
            legend('target','fit');
        hold off;
end

if (is_plot_forcing_term_fit)
    figure;
    subplot(2,1,1);
        hold on;
            plot([1:size(Fti,1)],Fti(:,1),'r');
            plot([1:size(Fti,1)],Fi(:,1),'b');
            title('Fi: target vs fit');
            legend('target','fit');
        hold off;
    subplot(2,1,2);
        hold on;
            plot([1:size(Fti,1)],Fti(:,2),'r');
            plot([1:size(Fti,1)],Fi(:,2),'b');
            title('Fi: target vs fit');
            legend('target','fit');
        hold off;
end

% dlmwrite('sample_traj_2D_unroll.txt', Yi, 'delimiter', ' ');

% end of Learning Baseline

%% Creating Synthetic Obstacle Avoidance Trajectory from Known Model

X_gTruth_cell   = cell(size(obs_list,2),1);
Yo_cell         = cell(size(obs_list,2),1);
Ydo_cell        = cell(size(obs_list,2),1);
Yddo_cell       = cell(size(obs_list,2),1);
gT_x3_cell      = cell(size(obs_list,2),1);
gT_v3_cell      = cell(size(obs_list,2),1);
Ct_SYNTH_cell   = cell(size(obs_list,2),1);

X_gTruth_stacked    = [];
gT_x3_stacked       = [];
gT_v3_stacked       = [];
Ct_SYNTH_stacked    = [];

for i=1:size(obs_list,2)
    [ X_gTruth_cell{i,1}, Yo_cell{i,1}, Ydo_cell{i,1}, Yddo_cell{i,1}, gT_x3_cell{i,1}, gT_v3_cell{i,1}, Ct_SYNTH_cell{i,1} ] = constructSynthObsAvoidCircleTraj( wi, w_SYNTH, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,i), traj_length, dt, beta_grid, k_grid, c_order, obs_radius );
    X_gTruth_stacked    = [X_gTruth_stacked; X_gTruth_cell{i,1}];
    gT_x3_stacked      = [gT_x3_stacked; gT_x3_cell{i,1}];
    gT_v3_stacked       = [gT_v3_stacked; gT_v3_cell{i,1}];
    Ct_SYNTH_stacked    = [Ct_SYNTH_stacked; Ct_SYNTH_cell{i,1}];
end

X_gTruth        = X_gTruth_cell{1,1};
Yo              = Yo_cell{1,1};
Ydo             = Ydo_cell{1,1};
Yddo            = Yddo_cell{1,1};
gT_x3           = gT_x3_cell{1,1};
gT_v3            = gT_v3_cell{1,1};
Ct_SYNTH        = Ct_SYNTH_cell{1,1};

if (is_plot_synth_obs_avoid_traj)
    figure;
    hold on;
    plot(Yi(:,1),Yi(:,2),'b');
    plot(Yo(:,1),Yo(:,2),'g');
    plot_circle(obs_list(1,1),obs_list(2,1), obs_radius, 'c');
    xlabel('x');
    ylabel('y');
    legend('baseline traj','obst avoid traj','obstacle');
    title('baseline vs obst avoid traj');
    hold off;

    keyboard;
end

% dlmwrite('synthetic_obst_avoid_traj.txt', Yo, 'delimiter', ' ');

% end of Creating Synthetic Obstacle Avoidance Trajectory from Known Model

%% Constructing Observed Obstacle Avoidance Features

X_observed_cell     = cell(size(obs_list,2),1);
T_x3_cell           = cell(size(obs_list,2),1);
T_v3_cell           = cell(size(obs_list,2),1);
Ct_target_cell      = cell(size(obs_list,2),1);
Ftarget_cell        = cell(size(obs_list,2),1);

X_observed_stacked      = [];
T_x3_stacked           = [];
T_v3_stacked            = [];
Ct_target_stacked       = [];

for i=1:size(obs_list,2)
    [ X_observed_cell{i,1}, T_x3_cell{i,1}, T_v3_cell{i,1} ] = constructObsAvoidCircleFeatMat2D( Yo_cell{i,1}, Ydo_cell{i,1}, obs_list(:,i), beta_grid, k_grid, obs_radius );
    [ Ct_target_cell{i,1}, Ftarget_cell{i,1} ] = computeDMPCtTarget( Yo_cell{i,1}, Ydo_cell{i,1}, Yddo_cell{i,1}, wi, n_rfs, sample_traj_2D(1,:)', sample_traj_2D(end,:)', dt, c_order );
    X_observed_stacked      = [X_observed_stacked; X_observed_cell{i,1}];
    T_x3_stacked           = [T_x3_stacked; T_x3_cell{i,1}];
    T_v3_stacked            = [T_v3_stacked; T_v3_cell{i,1}];
    Ct_target_stacked       = [Ct_target_stacked; Ct_target_cell{i,1}];
end

X_observed          = X_observed_cell{1,1};
T_x3                = T_x3_cell{1,1};
T_v3                = T_v3_cell{1,1};
Ct_target           = Ct_target_cell{1,1};
Ftarget             = Ftarget_cell{1,1};

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

% figure;
% hold on;
% subplot(2,2,1); plot(w_gT(:,1)); title('w (gTruth without ARD) d=1');
% subplot(2,2,2); plot(w_gT(:,2)); title('w (gTruth without ARD) d=2');
% subplot(2,2,3); plot(w_T(:,1)); title('w (observed without ARD) d=1');
% subplot(2,2,4); plot(w_T(:,2)); title('w (observed without ARD) d=2');
% hold off;

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

figure;
hold on;
subplot(2,2,1); plot(w_gT_SYNTH_ard(:,1)); title('w (X\_gTruth-Ct\_SYNTH with ARD) d=1');
subplot(2,2,2); plot(w_gT_SYNTH_ard(:,2)); title('w (X\_gTruth-Ct\_SYNTH with ARD) d=2');
subplot(2,2,3); plot(w_T_ard(:,1)); title('w (X\_observed-Ct\_target with ARD) d=1');
subplot(2,2,4); plot(w_T_ard(:,2)); title('w (X\_observed-Ct\_target with ARD) d=2');
hold off;

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

figure;
hold on;
subplot(2,2,1); plot(w_gT_SYNTH_lasso(:,1)); title('w (X\_gTruth-Ct\_SYNTH with LASSO) d=1');
subplot(2,2,2); plot(w_gT_SYNTH_lasso(:,2)); title('w (X\_gTruth-Ct\_SYNTH with LASSO) d=2');
subplot(2,2,3); plot(w_T_lasso(:,1)); title('w (X\_observed-Ct\_target with LASSO) d=1');
subplot(2,2,4); plot(w_T_lasso(:,2)); title('w (X\_observed-Ct\_target with LASSO) d=2');
hold off;

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

idx_limit_plot  = (size(Ct_SYNTH_stacked,1)/size(obs_list,2));

figure;
subplot(2,2,1);
    hold on;
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        plot(Ct_gT_fit(1:idx_limit_plot,1),Ct_gT_fit(1:idx_limit_plot,2),'m*');
        title('Ct synthetic vs Ct target vs Ct gTruth-fit, with Ridge Reg.');
        legend('synth','target','fit');
    hold off;
subplot(2,2,2);
    hold on;
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        plot(Ct_T_fit(1:idx_limit_plot,1),Ct_T_fit(1:idx_limit_plot,2),'m*');
        title('Ct synthetic vs Ct target vs Ct observed-fit, with Ridge Reg.');
        legend('synth','target','fit');
    hold off;
subplot(2,2,3); 
    hold on; 
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        plot(Ct_gT_fit_ard(1:idx_limit_plot,1),Ct_gT_fit_ard(1:idx_limit_plot,2),'m*');
        title('Ct synthetic vs Ct target vs Ct gTruth-fit, with ARD');
        legend('synth','target','fit');
    hold off;
subplot(2,2,4); 
    hold on; 
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        plot(Ct_T_fit_ard(1:idx_limit_plot,1),Ct_T_fit_ard(1:idx_limit_plot,2),'m*');
        title('Ct synthetic vs Ct target vs Ct observed-fit, with ARD');
        legend('synth','target','fit');
    hold off;

figure;
for d=1:2
    subplot(2,1,d);
        hold on;
            plot([1:idx_limit_plot],Ct_SYNTH(1:idx_limit_plot,d),'r');
            plot([1:idx_limit_plot],Ct_target(1:idx_limit_plot,d),'b');
            ftitle  = ['Ct synthetic vs Ct target, axis ',num2str(d)];
            title(ftitle);
            legend('synth','target');
        hold off;
end

% end of Comparison between Observed and Ground Truth, among Regression Methods

%% Comparison between Ground Truth vs Observed Variables

if (is_plot_gT_vs_obs_first)
    figure;
    hold on; subplot(2,2,1); plot([1:length(gT_x3(:,1))], gT_x3(:,1), 'b', [1:length(gT_x3(:,1))], T_x3(:,1), 'g'); legend('latent', 'observed'); title('d=1 (x3, x-axis)'); hold off;
    hold on; subplot(2,2,2); plot([1:length(gT_x3(:,1))], gT_x3(:,2), 'b', [1:length(gT_x3(:,1))], T_x3(:,2), 'g'); legend('latent', 'observed'); title('d=2 (x3, y-axis)'); hold off;
    hold on; subplot(2,2,3); plot([1:length(gT_x3(:,1))], gT_v3(:,1), 'b', [1:length(gT_x3(:,1))], T_v3(:,1), 'g'); legend('latent', 'observed'); title('d=1 (v3, x-axis)'); hold off;
    hold on; subplot(2,2,4); plot([1:length(gT_x3(:,1))], gT_v3(:,2), 'b', [1:length(gT_x3(:,1))], T_v3(:,2), 'g'); legend('latent', 'observed'); title('d=2 (v3, y-axis)'); hold off;
end
if (is_plot_gT_vs_obs_stacked)
    figure;
    hold on; subplot(2,2,1); plot([1:length(gT_x3_stacked(:,1))], gT_x3_stacked(:,1), 'b', [1:length(gT_x3_stacked(:,1))], T_x3_stacked(:,1), 'g'); legend('latent', 'observed'); title('d=1 (x3, x-axis)'); hold off;
    hold on; subplot(2,2,2); plot([1:length(gT_x3_stacked(:,1))], gT_x3_stacked(:,2), 'b', [1:length(gT_x3_stacked(:,1))], T_x3_stacked(:,2), 'g'); legend('latent', 'observed'); title('d=2 (x3, y-axis)'); hold off;
    hold on; subplot(2,2,3); plot([1:length(gT_x3_stacked(:,1))], gT_v3_stacked(:,1), 'b', [1:length(gT_x3_stacked(:,1))], T_v3_stacked(:,1), 'g'); legend('latent', 'observed'); title('d=1 (v3, x-axis)'); hold off;
    hold on; subplot(2,2,4); plot([1:length(gT_x3_stacked(:,1))], gT_v3_stacked(:,2), 'b', [1:length(gT_x3_stacked(:,1))], T_v3_stacked(:,2), 'g'); legend('latent', 'observed'); title('d=2 (v3, y-axis)'); hold off;
end