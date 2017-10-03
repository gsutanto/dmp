close                       all;
clear                       all;
clc;

addpath('../../utilities/');
addpath('../data/');

c_order                     = 1;

is_using_short_traj         = 0;
is_min_jerk                 = 0;
is_plot_fit_traj            = 1;
is_plot_forcing_term_fit    = 1;

%% Generate Obstacles

if (is_using_short_traj)
    n_rfs           = 43;
    nxgrid          = 20;
    nygrid          = 20;
    
    [obs_xgrid, obs_ygrid] = meshgrid(linspace(0.0,1.0,nxgrid),linspace(0.0,1.0,nygrid));
    obs_grid        = [reshape(obs_xgrid, 1, nxgrid*nygrid);...
                       reshape(obs_ygrid, 1, nxgrid*nygrid)];
                   
    % obs_list        = [[0.525,0.49]',obs_grid];
    % obs_list        = [0.525,0.49; 0.3,0.4; 0.75,0.4; 0.4, 0.45; 0.6, 0.475;...
    %                    0.8,0.35; 0.15,0.3; 0.225,0.35; 0.7,0.45; 0.85,0.3;...
    %                    0.45,0.475; 0.35,0.425; 0.475,0.45; 0.25,0.4]';
    obs_list        = [[0.525,0.49; 0.3,0.4; 0.75,0.4; 0.4, 0.45; 0.6, 0.475;...
                       0.8,0.35; 0.15,0.3; 0.225,0.35; 0.7,0.45; 0.85,0.3;...
                       0.45,0.475; 0.35,0.425; 0.475,0.45; 0.25,0.4]',obs_grid];
    % obs_list        = [0.525,0.49]';
else
    n_rfs           = 300;
    nxgrid          = 5;
    nygrid          = 5;

    [obs_xgrid, obs_ygrid] = meshgrid(linspace(0.0,2.0,nxgrid),linspace(0.0,2.0,nygrid));
    obs_grid        = [reshape(obs_xgrid, 1, nxgrid*nygrid);...
                       reshape(obs_ygrid, 1, nxgrid*nygrid)];
                   
    obs_list        = [[2.05,1.5]',obs_grid];
end

n_beta_grid     = 3;
n_k_grid        = 3;
% [beta_mgrid, k_mgrid] = meshgrid(linspace(1.0/pi,15.0/pi,n_beta_grid),linspace(0,15,n_k_grid));
[beta_mgrid, k_mgrid] = meshgrid(linspace(6.0/pi,14.0/pi,n_beta_grid),linspace(10,30,n_k_grid));
beta_grid       = reshape(beta_mgrid, n_beta_grid*n_k_grid, 1);
k_grid          = reshape(k_mgrid, n_beta_grid*n_k_grid, 1);
% beta            = 10/pi;  % default (from Peter Pastor's simulation)
% beta            = 4.5/pi; % Akshara's trial parameter
% k               = 3.75;   % Akshara's trial parameter
% beta            = 10/pi;  % synthetic data parameter
% k               = 20;     % synthetic data parameter
% sigsq           = 100.0;  % parameter for Gaussian Lyapunov basis funct
% betaDYN2        = 0.5;    % parameter for DYN2 Lyapunov basis funct
% kDYN2           = 10.0;   % parameter for DYN2 Lyapunov basis funct

w_SYNTH         = zeros(2*n_beta_grid*n_k_grid,2);

if (is_using_short_traj)
    w_SYNTH(n_beta_grid*n_k_grid,1)   = 70;
    w_SYNTH(n_beta_grid*n_k_grid+1,2) = 70;
else
    w_SYNTH(n_beta_grid*n_k_grid,1)   = 1500;
    w_SYNTH(n_beta_grid*n_k_grid+1,2) = 500;
end

% end of Generate Obstacles

%% Load the Data

if (is_min_jerk)
    tau_min_jerk    = 0.5;
    dt              = 0.001;
    for d=1:2
        [sample_traj_2D(:,d), sample_traj_2D_d(:,d), sample_traj_2D_dd(:,d)] = dcp_franzi('generate_minjerk',tau_min_jerk,dt);
    end
else
    if (is_using_short_traj)
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
    else
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
end

traj_length             = size(sample_traj_2D,1);

% data            = cell(1,3);
% data{1,1}       = cell(1,1);
% data{1,1}{1,1}  = sample_traj_2D;
% data{1,2}       = obs_list[1,:];
% data{1,3}{1,1}  = Yo;

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

X_latent_cell   = cell(size(obs_list,2),1);
Yo_cell         = cell(size(obs_list,2),1);
Ydo_cell        = cell(size(obs_list,2),1);
Yddo_cell       = cell(size(obs_list,2),1);
L_ox3_cell      = cell(size(obs_list,2),1);
L_v3_cell       = cell(size(obs_list,2),1);
Ct_SYNTH_cell   = cell(size(obs_list,2),1);

X_latent_stacked    = [];
L_ox3_stacked       = [];
L_v3_stacked        = [];
Ct_SYNTH_stacked    = [];

for i=1:size(obs_list,2)
    [ X_latent_cell{i,1}, Yo_cell{i,1}, Ydo_cell{i,1}, Yddo_cell{i,1}, L_ox3_cell{i,1}, L_v3_cell{i,1}, Ct_SYNTH_cell{i,1} ] = constructSynthObsAvoidPointTraj( wi, w_SYNTH, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,i), traj_length, dt, beta_grid, k_grid, c_order );
    X_latent_stacked    = [X_latent_stacked; X_latent_cell{i,1}];
    L_ox3_stacked       = [L_ox3_stacked; L_ox3_cell{i,1}];
    L_v3_stacked        = [L_v3_stacked; L_v3_cell{i,1}];
    Ct_SYNTH_stacked    = [Ct_SYNTH_stacked; Ct_SYNTH_cell{i,1}];
end

X_latent        = X_latent_cell{1,1};
Yo              = Yo_cell{1,1};
Ydo             = Ydo_cell{1,1};
Yddo            = Yddo_cell{1,1};
L_ox3           = L_ox3_cell{1,1};
L_v3            = L_v3_cell{1,1};
Ct_SYNTH        = Ct_SYNTH_cell{1,1};

% figure;
% hold on;
% plot(Yi(:,1),Yi(:,2),'b');
% plot(Yo(:,1),Yo(:,2),'g');
% scatter(obs_list(1,1),obs_list(2,1),'co');
% xlabel('x');
% ylabel('y');
% legend('baseline traj','obst avoid traj','obstacle');
% title('baseline vs obst avoid traj');
% hold off;
% 
% keyboard;

% dlmwrite('synthetic_obst_avoid_traj.txt', Yo, 'delimiter', ' ');

% end of Creating Synthetic Obstacle Avoidance Trajectory from Known Model

%% Learning Synthetic Obstacle Avoidance Trajectory

wo_cell         = cell(size(obs_list,2),1);
Yofit_cell      = cell(size(obs_list,2),1);
Ydofit_cell     = cell(size(obs_list,2),1);
Yddofit_cell    = cell(size(obs_list,2),1);
Fofit_cell      = cell(size(obs_list,2),1);
Fto_cell        = cell(size(obs_list,2),1);

for i=1:size(obs_list,2)
    [ wo_cell{i,1}, Yofit_cell{i,1}, Ydofit_cell{i,1}, Yddofit_cell{i,1}, Fto_cell{i,1}, Fofit_cell{i,1} ] = learnPrimitive( Yo_cell{i,1}, Ydo_cell{i,1}, Yddo_cell{i,1}, n_rfs, sample_traj_2D(1,:), sample_traj_2D(end,:), traj_length, dt, c_order );
end

wo              = wo_cell{1,1};

if (is_plot_fit_traj)
    figure;
    i=1;
    subplot(2,2,1);
        hold on;
            plot(Yo_cell{i,1}(:,1),Yo_cell{i,1}(:,2),'r');
            plot(Yofit_cell{i,1}(:,1),Yofit_cell{i,1}(:,2),'b');
            title('Yo: data vs fit');
            legend('data','fit');
        hold off;
    subplot(2,2,2);
        hold on;
            plot(Ydo_cell{i,1}(:,1),Ydo_cell{i,1}(:,2),'r');
            plot(Ydofit_cell{i,1}(:,1),Ydofit_cell{i,1}(:,2),'b');
            title('Ydo: data vs fit');
            legend('data','fit');
        hold off;
    subplot(2,2,3);
        hold on;
            plot(Yddo_cell{i,1}(:,1),Yddo_cell{i,1}(:,2),'r');
            plot(Yddofit_cell{i,1}(:,1),Yddofit_cell{i,1}(:,2),'b');
            title('Yddo: data vs fit');
            legend('data','fit');
        hold off;
    subplot(2,2,4);
        hold on;
            plot(Fto_cell{i,1}(:,1),Fto_cell{i,1}(:,2),'r');
            plot(Fofit_cell{i,1}(:,1),Fofit_cell{i,1}(:,2),'b');
            title('Fo: target vs fit');
            legend('target','fit');
        hold off;
end

if (is_plot_forcing_term_fit)
    figure;
    subplot(2,1,1);
        hold on;
            plot([1:size(Fto_cell{i,1},1)],Fto_cell{i,1}(:,1),'r');
            plot([1:size(Fto_cell{i,1},1)],Fofit_cell{i,1}(:,1),'b');
            title('Fo: target vs fit');
            legend('target','fit');
        hold off;
    subplot(2,1,2);
        hold on;
            plot([1:size(Fto_cell{i,1},1)],Fto_cell{i,1}(:,2),'r');
            plot([1:size(Fto_cell{i,1},1)],Fofit_cell{i,1}(:,2),'b');
            title('Fo: target vs fit');
            legend('target','fit');
        hold off;
end

% end of Learning Synthetic Obstacle Avoidance Trajectory

%% Constructing Observed Obstacle Avoidance Features

X_observed_cell     = cell(size(obs_list,2),1);
Fo_cell             = cell(size(obs_list,2),1);
You_cell            = cell(size(obs_list,2),1);
Ydou_cell           = cell(size(obs_list,2),1);
Yddou_cell          = cell(size(obs_list,2),1);
T_ox3_cell          = cell(size(obs_list,2),1);
T_v3_cell           = cell(size(obs_list,2),1);
Ct_target_cell      = cell(size(obs_list,2),1);
Ct_SYN_recons_cell  = cell(size(obs_list,2),1);
Ftarget_cell        = cell(size(obs_list,2),1);
FSYN_recons_cell    = cell(size(obs_list,2),1);

Fo_stacked              = [];
X_observed_stacked      = [];
T_ox3_stacked           = [];
T_v3_stacked            = [];
Ct_target_stacked       = [];
Ct_SYN_recons_stacked   = [];

for i=1:size(obs_list,2)
    [ X_observed_cell{i,1}, Fo_cell{i,1}, You_cell{i,1}, Ydou_cell{i,1}, Yddou_cell{i,1}, T_ox3_cell{i,1}, T_v3_cell{i,1} ] = constructObsAvoidPointFeatMat2D_old( wo_cell{i,1}, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,i), traj_length, dt, beta_grid, k_grid, c_order );
    [ Ct_target_cell{i,1}, Ftarget_cell{i,1} ]          = computeDMPCtTarget( You_cell{i,1}, Ydou_cell{i,1}, Yddou_cell{i,1}, wi, n_rfs, sample_traj_2D(1,:)', sample_traj_2D(end,:)', dt, c_order );
    [ Ct_SYN_recons_cell{i,1}, FSYN_recons_cell{i,1} ]  = computeDMPCtTarget( Yo_cell{i,1}, Ydo_cell{i,1}, Yddo_cell{i,1}, wi, n_rfs, sample_traj_2D(1,:)', sample_traj_2D(end,:)', dt, c_order );
    Fo_stacked              = [Fo_stacked; Fo_cell{i,1}];
    X_observed_stacked      = [X_observed_stacked; X_observed_cell{i,1}];
    T_ox3_stacked           = [T_ox3_stacked; T_ox3_cell{i,1}];
    T_v3_stacked            = [T_v3_stacked; T_v3_cell{i,1}];
    Ct_target_stacked       = [Ct_target_stacked; Ct_target_cell{i,1}];
    Ct_SYN_recons_stacked   = [Ct_SYN_recons_stacked; Ct_SYN_recons_cell{i,1}];
end

X_observed          = X_observed_cell{1,1};
Fo                  = Fo_cell{1,1};
You                 = You_cell{1,1};
T_ox3               = T_ox3_cell{1,1};
T_v3                = T_v3_cell{1,1};
Ct_target           = Ct_target_cell{1,1};
Ct_SYN_recons       = Ct_SYN_recons_cell{1,1};
Ftarget             = Ftarget_cell{1,1};
FSYN_recons         = FSYN_recons_cell{1,1};

figure;
hold on;
plot(Yi(:,1),Yi(:,2),'b');
plot(You(:,1),You(:,2),'g');
scatter(obs_list(1,1),obs_list(2,1),'co');
xlabel('x');
ylabel('y');
legend('DMP-unrolled baseline (Yi)','DMP-unrolled synthetic obst avoid demo (You)','obstacle');
title('DMP-unrolled baseline vs synthetic obst avoid demonstration');
hold off;

% Constructing Observed Obstacle Avoidance Features

%% Regression (without ARD)

Fi_stacked  = repmat(Fi,size(obs_list,2),1);
Fo_minus_Fi = Fo_stacked - Fi_stacked; % turns out equal to Ct_target, as computed above/before...

XX_L        = X_latent_stacked.'*X_latent_stacked;
XX_T        = X_observed_stacked.'*X_observed_stacked;

% xc_L        = X_latent_stacked.'*Fo_minus_Fi;
xc_L_SYNTH  = X_latent_stacked.'*Ct_SYNTH_stacked;
xc_L        = X_latent_stacked.'*Ct_target_stacked;
% xc_T        = X_observed_stacked.'*Fo_minus_Fi;
xc_T_SYNTH  = X_observed_stacked.'*Ct_SYNTH_stacked;
xc_T        = X_observed_stacked.'*Ct_target_stacked;

reg         = 1e-9;
A           = reg*eye(size(XX_L,2));

w_L_SYNTH   = (A + XX_L)\xc_L_SYNTH;
w_T_SYNTH   = (A + XX_L)\xc_T_SYNTH;

w_L         = (A + XX_L)\xc_L;
w_T         = (A + XX_T)\xc_T;

% figure;
% hold on;
% subplot(2,2,1); plot(w_L(:,1)); title('w (latent without ARD) d=1');
% subplot(2,2,2); plot(w_L(:,2)); title('w (latent without ARD) d=2');
% subplot(2,2,3); plot(w_T(:,1)); title('w (observed without ARD) d=1');
% subplot(2,2,4); plot(w_T(:,2)); title('w (observed without ARD) d=2');
% hold off;

% end of Regression (without ARD)

%% Unrolling (Observed, without ARD)

[ Yt, Ytd, Ytdd, Ft ] = unrollObsAvoidPointTraj( wi, w_T, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,1), traj_length, dt, beta_grid, k_grid, c_order );

% end of Unrolling (Observed, without ARD)

%% Unrolling (Latent, without ARD)

[ Yl, Yld, Yldd, Fl ] = unrollObsAvoidPointTraj( wi, w_L, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,1), traj_length, dt, beta_grid, k_grid, c_order );

% end of Unrolling (Latent, without ARD)

%% Comparison between Observed and Latent (without ARD)

% figure;
% hold on;
% plot(Yi(:,1),Yi(:,2),'b');
% plot(You(:,1),You(:,2),'g');
% plot(Yt(:,1),Yt(:,2),'m');
% plot(Yl(:,1),Yl(:,2),'k');
% scatter(obs_list(1,1),obs_list(2,1),'co');
% xlabel('x');
% ylabel('y');
% legend('DMP-unrolled baseline','DMP-unrolled synthetic obst avoid demo',...
%        'DMP-unrolled learned obst avoid (train)','DMP-unrolled learned obst avoid (IF latent variable is known)',...
%        'obstacle');
% title('DMP-unrolled baseline vs synthetic obst avoid demo vs learned obst avoid (train) vs learned obst avoid (latent variable known/used) without ARD');
% hold off;

% end of Comparison between Observed and Latent (without ARD)

%% Regression (with ARD)
w_L_SYNTH_ard   = zeros(size(w_L));
w_T_SYNTH_ard   = zeros(size(w_T));
w_L_ard         = zeros(size(w_L));
w_T_ard         = zeros(size(w_T));
for d = 1:2
    display(['ARD dim: ', num2str(d)]);
   
%    [w_L_ard_d,r_L_idx] = ARD(X_latent_stacked,Fo_minus_Fi(:,d), 0);
   
    [w_L_SYNTH_ard_d,r_L_SYNTH_idx] = ARD(X_latent_stacked,Ct_SYNTH_stacked(:,d), 0);
    w_L_SYNTH_ard(r_L_SYNTH_idx,d)  = w_L_SYNTH_ard_d;
   
    [w_L_ard_d,r_L_idx]  = ARD(X_latent_stacked,Ct_target_stacked(:,d), 0);
    w_L_ard(r_L_idx,d)   = w_L_ard_d;
   
%    [w_T_ard_d,r_T_idx] = ARD(X_observed_stacked,Fo_minus_Fi(:,d), 0);
   
    [w_T_SYNTH_ard_d,r_T_SYNTH_idx] = ARD(X_observed_stacked,Ct_SYNTH_stacked(:,d), 0);
    w_T_SYNTH_ard(r_T_SYNTH_idx,d)  = w_T_SYNTH_ard_d;
    
    [w_T_ard_d,r_T_idx] = ARD(X_observed_stacked,Ct_target_stacked(:,d), 0);
    w_T_ard(r_T_idx,d)  = w_T_ard_d;
end

figure;
hold on;
subplot(2,2,1); plot(w_L_SYNTH_ard(:,1)); title('w (X\_latent-Ct\_SYNTH with ARD) d=1');
subplot(2,2,2); plot(w_L_SYNTH_ard(:,2)); title('w (X\_latent-Ct\_SYNTH with ARD) d=2');
subplot(2,2,3); plot(w_T_SYNTH_ard(:,1)); title('w (X\_observed-Ct\_SYNTH with ARD) d=1');
subplot(2,2,4); plot(w_T_SYNTH_ard(:,2)); title('w (X\_observed-Ct\_SYNTH with ARD) d=2');
hold off;

figure;
hold on;
subplot(2,2,1); plot(w_L_ard(:,1)); title('w (X\_latent-Ct\_target with ARD) d=1');
subplot(2,2,2); plot(w_L_ard(:,2)); title('w (X\_latent-Ct\_target with ARD) d=2');
subplot(2,2,3); plot(w_T_ard(:,1)); title('w (X\_observed-Ct\_target with ARD) d=1');
subplot(2,2,4); plot(w_T_ard(:,2)); title('w (X\_observed-Ct\_target with ARD) d=2');
hold off;

% end of Regression (with ARD)

%% Unrolling (Observed, with ARD)

[ YSA_T, YdSA_T, YddSA_T, FSA_T ] = unrollObsAvoidPointTraj( wi, w_T_SYNTH_ard, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,1), traj_length, dt, beta_grid, k_grid, c_order );
[ YA_T, YdA_T, YddA_T, FA_T ] = unrollObsAvoidPointTraj( wi, w_T_ard, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,1), traj_length, dt, beta_grid, k_grid, c_order );

% end of Unrolling (Observed, with ARD)

%% Unrolling (Latent, with ARD)

[ YSA_L, YdSA_L, YddSA_L, FSA_L ] = unrollObsAvoidPointTraj( wi, w_L_SYNTH_ard, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,1), traj_length, dt, beta_grid, k_grid, c_order );
[ YA_L, YdA_L, YddA_L, FA_L ] = unrollObsAvoidPointTraj( wi, w_L_ard, sample_traj_2D(1,:), sample_traj_2D(end,:), obs_list(:,1), traj_length, dt, beta_grid, k_grid, c_order );

% end of Unrolling (Latent, with ARD)

%% Comparison between Observed and Latent (with ARD)

fprintf('rank(XX_L) = %d\n', rank(XX_L));
fprintf('rank(XX_T) = %d\n', rank(XX_T));

% [mse_L,nmse_L]  = computeNMSE(X_latent_stacked, w_L, Fo_minus_Fi);
[mseS_L,nmseS_L]  = computeNMSE(X_latent_stacked, w_L_SYNTH, Ct_SYNTH_stacked);
[mse_L,nmse_L]  = computeNMSE(X_latent_stacked, w_L, Ct_target_stacked);
% [mse_T,nmse_T]  = computeNMSE(X_observed_stacked, w_T, Fo_minus_Fi);
[mseS_T,nmseS_T]  = computeNMSE(X_observed_stacked, w_T_SYNTH, Ct_SYNTH_stacked);
[mse_T,nmse_T]  = computeNMSE(X_observed_stacked, w_T, Ct_target_stacked);

% [mse_L_ard,nmse_L_ard]  = computeNMSE(X_latent_stacked, w_L_ard, Fo_minus_Fi);
[mseS_L_ard,nmseS_L_ard]  = computeNMSE(X_latent_stacked, w_L_SYNTH_ard, Ct_SYNTH_stacked);
[mse_L_ard,nmse_L_ard]  = computeNMSE(X_latent_stacked, w_L_ard, Ct_target_stacked);
% [mse_T_ard,nmse_T_ard]  = computeNMSE(X_observed_stacked, w_T_ard, Fo_minus_Fi);
[mseS_T_ard,nmseS_T_ard]  = computeNMSE(X_observed_stacked, w_T_SYNTH_ard, Ct_SYNTH_stacked);
[mse_T_ard,nmse_T_ard]  = computeNMSE(X_observed_stacked, w_T_ard, Ct_target_stacked);

disp(['nmse (X_latent-Ct_SYNTH, no ARD)    = ', num2str(nmseS_L)]);
disp(['nmse (X_observed-Ct_SYNTH, no ARD)  = ', num2str(nmseS_T)]);
disp(['nmse (X_latent-Ct_SYNTH, w/ ARD)    = ', num2str(nmseS_L_ard)]);
disp(['nmse (X_observed-Ct_SYNTH, w/ ARD)  = ', num2str(nmseS_T_ard)]);
disp(['nmse (X_latent-Ct_target, no ARD)    = ', num2str(nmse_L)]);
disp(['nmse (X_observed-Ct_target, no ARD)  = ', num2str(nmse_T)]);
disp(['nmse (X_latent-Ct_target, w/ ARD)    = ', num2str(nmse_L_ard)]);
disp(['nmse (X_observed-Ct_target, w/ ARD)  = ', num2str(nmse_T_ard)]);

% figure;
% hold on;
% plot(Yi(:,1),Yi(:,2),'b');
% plot(You(:,1),You(:,2),'g');
% plot(YA_T(:,1),YA_T(:,2),'m');
% plot(YA_L(:,1),YA_L(:,2),'k');
% scatter(obs_list(1,1),obs_list(2,1),'co');
% xlabel('x');
% ylabel('y');
% legend('DMP-unrolled baseline','DMP-unrolled synthetic obst avoid demo',...
%        'DMP-unrolled learned obst avoid (train)','DMP-unrolled learned obst avoid (IF latent variable is known)',...
%        'obstacle');
% title('DMP-unrolled baseline vs synthetic obst avoid demo vs learned obst avoid (train) vs learned obst avoid (latent variable known/used) with ARD');
% hold off;

Ct_L_fit        = X_latent_stacked * w_L;
Ct_L_fit_ard    = X_latent_stacked * w_L_ard;
Ct_T_fit        = X_observed_stacked * w_T;
Ct_T_fit_ard    = X_observed_stacked * w_T_ard;

idx_limit_plot  = (size(Fo_minus_Fi,1)/size(obs_list,2));

% figure;
% subplot(2,1,1); 
%     hold on; 
%         plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
%         plot(Ct_L_fit_ard(1:idx_limit_plot,1),Ct_L_fit_ard(1:idx_limit_plot,2),'m');
%         plot(Fo_minus_Fi(1:idx_limit_plot,1),Fo_minus_Fi(1:idx_limit_plot,2),'g');
%         title('Ct synthetic vs Ct synth-fit (latent) vs (Fo-Fi), with ARD');
%         legend('synth','fit','Fo-Fi');
%     hold off;
% subplot(2,1,2); 
%     hold on; 
%         plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
%         plot(Ct_T_fit_ard(1:idx_limit_plot,1),Ct_T_fit_ard(1:idx_limit_plot,2),'m');
%         plot(Fo_minus_Fi(1:idx_limit_plot,1),Fo_minus_Fi(1:idx_limit_plot,2),'g');
%         title('Ct synthetic vs Ct synth-fit (observed) vs (Fo-Fi), with ARD');
%         legend('synth','fit','Fo-Fi');
%     hold off;

figure;
subplot(2,2,1);
    hold on;
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_SYN_recons(1:idx_limit_plot,1),Ct_SYN_recons(1:idx_limit_plot,2),'rx');
        plot(Ct_L_fit(1:idx_limit_plot,1),Ct_L_fit(1:idx_limit_plot,2),'m*');
        plot(Fo_minus_Fi(1:idx_limit_plot,1),Fo_minus_Fi(1:idx_limit_plot,2),'go');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        title('Ct synthetic vs Ct synth-fit (latent) vs (Fo-Fi) vs Ct target, no ARD');
        legend('synth','reconst synth','fit','Fo-Fi','target');
    hold off;
subplot(2,2,2);
    hold on;
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_SYN_recons(1:idx_limit_plot,1),Ct_SYN_recons(1:idx_limit_plot,2),'rx');
        plot(Ct_T_fit(1:idx_limit_plot,1),Ct_T_fit(1:idx_limit_plot,2),'m*');
        plot(Fo_minus_Fi(1:idx_limit_plot,1),Fo_minus_Fi(1:idx_limit_plot,2),'go');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        title('Ct synthetic vs Ct synth-fit (observed) vs (Fo-Fi) vs Ct target, no ARD');
        legend('synth','reconst synth','fit','Fo-Fi','target');
    hold off;
subplot(2,2,3); 
    hold on; 
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_SYN_recons(1:idx_limit_plot,1),Ct_SYN_recons(1:idx_limit_plot,2),'rx');
        plot(Ct_L_fit_ard(1:idx_limit_plot,1),Ct_L_fit_ard(1:idx_limit_plot,2),'m*');
        plot(Fo_minus_Fi(1:idx_limit_plot,1),Fo_minus_Fi(1:idx_limit_plot,2),'go');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        title('Ct synthetic vs Ct synth-fit (latent) vs (Fo-Fi) vs Ct target, with ARD');
        legend('synth','reconst synth','fit','Fo-Fi','target');
    hold off;
subplot(2,2,4); 
    hold on; 
        plot(Ct_SYNTH(1:idx_limit_plot,1),Ct_SYNTH(1:idx_limit_plot,2),'r');
        plot(Ct_SYN_recons(1:idx_limit_plot,1),Ct_SYN_recons(1:idx_limit_plot,2),'rx');
        plot(Ct_T_fit_ard(1:idx_limit_plot,1),Ct_T_fit_ard(1:idx_limit_plot,2),'m*');
        plot(Fo_minus_Fi(1:idx_limit_plot,1),Fo_minus_Fi(1:idx_limit_plot,2),'go');
        plot(Ct_target(1:idx_limit_plot,1),Ct_target(1:idx_limit_plot,2),'b');
        title('Ct synthetic vs Ct synth-fit (observed) vs (Fo-Fi) vs Ct target, with ARD');
        legend('synth','reconst synth','fit','Fo-Fi','target');
    hold off;

figure;
for d=1:2
    subplot(2,1,d);
        hold on;
            plot([1:idx_limit_plot],Ct_SYNTH(1:idx_limit_plot,d),'b');
%             plot([1:idx_limit_plot],Ct_SYN_recons(1:idx_limit_plot,d),'rx');
            plot([1:idx_limit_plot],Fo_minus_Fi(1:idx_limit_plot,d),'g');
%             plot([1:idx_limit_plot],Ct_target(1:idx_limit_plot,d),'b');
%             ftitle  = ['Ct synthetic vs (Fo-Fi) vs Ct target vs Reconstructed Ct synthetic, axis ',num2str(d)];
            ftitle  = ['Ct synthetic vs (Fo-Fi) with respect to time, axis ',num2str(d)];
            title(ftitle);
%             legend('synth','reconst synth','Fo-Fi','target');
            legend('synth','Fo-Fi');
        hold off;
end

% end of Comparison between Observed and Latent (with ARD)

%% Regression (with LASSO)
% w_T_lasso       = zeros(size(w_T));
% lasso_result    = cell(2,1);
% for d=1:2
%      lasso_result{d,1}  = lasso(X_observed_stacked,Fo_minus_Fi(:,d));
%      w_T_lasso(:,d)     = lasso_result{d,1}(:,round(size(lasso_result{d,1})/2));
% end

% figure;
% hold on;
% subplot(2,1,1); plot(w_T_lasso(:,1)); title('w (observed with LASSO) d=1');
% subplot(2,1,2); plot(w_T_lasso(:,2)); title('w (observed with LASSO) d=2');
% hold off;

% end of Regression (with LASSO)

%% Unrolling (Observed, with LASSO)



% end of Unrolling (Observed, with LASSO)

%% Regression (Positivity Constraint)

% w_L_PC      = zeros(size(xc_L));
% w_T_PC      = zeros(size(xc_T));
% for d=1:2
%     w_L_PC(:,d)=quadprog((XX_L),-2*xc_L(:,d),[],[],[],[],zeros(size(xc_L,1),1),Inf(size(xc_L,1),1));
%     w_T_PC(:,d)=quadprog((XX_T),-2*xc_T(:,d),[],[],[],[],zeros(size(xc_T,1),1),Inf(size(xc_T,1),1));
% end
% 
% figure;
% hold on;
% subplot(2,2,1); plot(w_L_PC(:,1)); title('w (latent, positivity constraint) d=1');
% subplot(2,2,2); plot(w_L_PC(:,2)); title('w (latent, positivity constraint) d=2');
% subplot(2,2,3); plot(w_T_PC(:,1)); title('w (observed, positivity constraint) d=1');
% subplot(2,2,4); plot(w_T_PC(:,2)); title('w (observed, positivity constraint) d=2');
% hold off;

% end of Regression (without ARD)

%% Comparison between Logged Latend vs Observed Variables

% figure;
% hold on; subplot(2,2,1); plot([1:length(L_ox3_stacked(:,1))], L_ox3_stacked(:,1), 'b', [1:length(L_ox3_stacked(:,1))], T_ox3_stacked(:,1), 'g'); legend('latent', 'observed'); title('d=1 (ox3, x-axis)'); hold off;
% hold on; subplot(2,2,2); plot([1:length(L_ox3_stacked(:,1))], L_ox3_stacked(:,2), 'b', [1:length(L_ox3_stacked(:,1))], T_ox3_stacked(:,2), 'g'); legend('latent', 'observed'); title('d=2 (ox3, y-axis)'); hold off;
% hold on; subplot(2,2,3); plot([1:length(L_ox3_stacked(:,1))], L_v3_stacked(:,1), 'b', [1:length(L_ox3_stacked(:,1))], T_v3_stacked(:,1), 'g'); legend('latent', 'observed'); title('d=1 (v3, x-axis)'); hold off;
% hold on; subplot(2,2,4); plot([1:length(L_ox3_stacked(:,1))], L_v3_stacked(:,2), 'b', [1:length(L_ox3_stacked(:,1))], T_v3_stacked(:,2), 'g'); legend('latent', 'observed'); title('d=2 (v3, y-axis)'); hold off;