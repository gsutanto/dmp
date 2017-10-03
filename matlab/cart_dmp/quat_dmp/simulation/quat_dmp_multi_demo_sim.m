% Quaternion Dynamic Movement Primitive (DMP) Simulation
% -Learning from Multiple Quaternion Trajectory Demonstrations-
% 
% Author : Giovanni Sutanto
% Date   : December 2016

clear all;
close all;
clc;

addpath('../../../utilities/');
addpath('../');

D   = 3;
Dq  = 4;

data_path           = '../../../../data/cart_dmp/cart_dmp_wiggling/';

%% Gold Quaternion Trajectory Demonstration:
% Load Human Pose Quaternion Trajectory Demonstration:
cart_Q_Qd_Qdd_traj_demo_gold        = dlmread([data_path, 'sample_quat_traj_recorded_demo_wiggling.txt']);
cart_Q_omega_omegad_traj_demo_gold  = dlmread([data_path, 'sample_quat_ABGomega_traj_recorded_demo_wiggling.txt']);
time_gold   = cart_Q_Qd_Qdd_traj_demo_gold(:,1)';
QT_gold     = cart_Q_Qd_Qdd_traj_demo_gold(:,2:5)';
QdT_gold    = cart_Q_Qd_Qdd_traj_demo_gold(:,6:9)';
QddT_gold   = cart_Q_Qd_Qdd_traj_demo_gold(:,10:13)';

% extracting/converting omega and omegad (trajectories) 
% from trajectories of Q, Qd, and Qdd
% [ omegaT_gold, omegadT_gold ] = computeOmegaAndOmegaDotTrajectory( QT_gold, QdT_gold, QddT_gold );
omegaT_gold = cart_Q_omega_omegad_traj_demo_gold(:,6:8)';
omegadT_gold= cart_Q_omega_omegad_traj_demo_gold(:,9:11)';

tau_gold            = time_gold(1,end) - time_gold(1,1);
traj_length_gold    = size(QT_gold,2);
dt_gold             = tau_gold/(traj_length_gold-1);

% Initial Orientation Quaternion:
Q0_gold             = QT_gold(:,1);
Q0_gold             = Q0_gold/norm(Q0_gold);

% Goal Orientation Quaternion:
QG_gold             = QT_gold(:,end);
QG_gold             = QG_gold/norm(QG_gold);

%% Multiple (Perturbed) Quaternion Trajectory Demonstrations:
multi_demo_data_path= [data_path, '/dummy_samples_quat_traj_demo_wiggling/'];

% Load Perturbed Human Pose Quaternion Trajectory Demonstrations:
Quat_traj_demo_set = cell(3, 1);
dts             = [];
Q0s             = [];
QGs             = [];
file_idx        = 1;
while (exist([multi_demo_data_path, num2str(file_idx), '.txt'], 'file') == 2)
    cart_Q_omega_omegad_traj_demo   = dlmread([multi_demo_data_path, num2str(file_idx), '.txt']);
    time        = cart_Q_omega_omegad_traj_demo(:,1)';
    QT          = cart_Q_omega_omegad_traj_demo(:,2:5)';
    omegaT      = cart_Q_omega_omegad_traj_demo(:,6:8)';
    omegadT     = cart_Q_omega_omegad_traj_demo(:,9:11)';

    tau         = time(1,end) - time(1,1);
    traj_length = size(QT,2);
    dt          = tau/(traj_length-1);

    % Initial Orientation Quaternion:
    Q0      = QT(:,1);
    Q0      = Q0/norm(Q0);

    % Goal Orientation Quaternion:
    QG      = QT(:,end);
    QG      = QG/norm(QG);
    
    Quat_traj_demo_set{1, file_idx} = QT;
    Quat_traj_demo_set{2, file_idx} = omegaT;
    Quat_traj_demo_set{3, file_idx} = omegadT;
    dts         = [dts, dt];
    Q0s         = [Q0s, Q0];
    QGs         = [QGs, QG];
    
    file_idx    = file_idx + 1;
end
N_demo          = file_idx - 1;
clearvars       cart_Q_omega_omegad_traj_demo time QT omegaT omegadT tau traj_length dt Q0 QG file_idx;
fprintf('Loaded %d Quaternion trajectory demonstrations.\n', N_demo);

assert(var(dts) < 1e-10, 'Sampling Time (dt) is inconsistent across demonstrated trajectories.');
Q0s     = normalizeQuaternion(Q0s);
QGs     = normalizeQuaternion(QGs);
mean_dt = mean(dts);
mean_Q0 = standardizeNormalizeQuaternion(computeAverageQuaternions(Q0s));
mean_QG = standardizeNormalizeQuaternion(computeAverageQuaternions(QGs));

%% Quaternion DMP Parameter Setup
global  dcps;

n_rfs   = 25;
c_order = 1;
w     	= zeros(n_rfs,3);
ID      = 1;

%% Fitting/Learning the Quaternion DMP based on Dataset
dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
dcp_quaternion('reset_state', ID, mean_Q0);
dcp_quaternion('set_goal', ID, mean_QG, 1);

[w, F_target, F_fit] = dcp_quaternion('batch_fit_multi', ID, mean_dt, Quat_traj_demo_set);
dlmwrite([data_path, '/weights_quat_dmp_wiggling.txt'], w, 'delimiter', ' ');

%% Unrolling based on Dataset (using mean_Q0 and mean_QG)
Q_hist      = zeros(Dq, traj_length_gold);
Qd_hist     = zeros(Dq, traj_length_gold);
Qdd_hist    = zeros(Dq, traj_length_gold);
omega_hist  = zeros(D, traj_length_gold);
omegad_hist = zeros(D, traj_length_gold);
F_run       = zeros(D, traj_length_gold);

dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
dcp_quaternion('reset_state', ID, mean_Q0);
dcp_quaternion('set_goal', ID, mean_QG, 1);
dcps(1).w   = w;

for i=1:traj_length_gold
    [Q, Qd, Qdd, omega, omegad, f] = dcp_quaternion('run', ID, tau_gold, mean_dt);

    Q_hist(:,i)     = Q;
    Qd_hist(:,i)    = Qd;
    Qdd_hist(:,i)   = Qdd;
    omega_hist(:,i) = omega;
    omegad_hist(:,i)= omegad;

    F_run(:,i)      = f;
end

%% Unrolling based on Gold/Ground Truth Contexts (using Q0_gold and QG_gold)
Q_hist_gc       = zeros(Dq, traj_length_gold);
Qd_hist_gc      = zeros(Dq, traj_length_gold);
Qdd_hist_gc     = zeros(Dq, traj_length_gold);
omega_hist_gc   = zeros(D, traj_length_gold);
omegad_hist_gc  = zeros(D, traj_length_gold);
F_run_gc        = zeros(D, traj_length_gold);

dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
dcp_quaternion('reset_state', ID, Q0_gold);
dcp_quaternion('set_goal', ID, QG_gold, 1);
dcps(1).w   = w;

for i=1:traj_length_gold
    [Q_gc, Qd_gc, Qdd_gc, omega_gc, omegad_gc, f_gc] = dcp_quaternion('run', ID, tau_gold, dt_gold);

    Q_hist_gc(:,i)      = Q_gc;
    Qd_hist_gc(:,i)     = Qd_gc;
    Qdd_hist_gc(:,i)    = Qdd_gc;
    omega_hist_gc(:,i)  = omega_gc;
    omegad_hist_gc(:,i) = omegad_gc;

    F_run_gc(:,i)       = f_gc;
end

%% Plotting Unrolling Results versus Gold/Unperturbed Trajectory
is_plotting_vs_gold_traj    = 0;

if (is_plotting_vs_gold_traj)
    figure;
    for d=1:D
        subplot(D,1,d);
            hold on;
                plot(F_run_gc(d,:),'g');
                plot(F_run(d,:),'b');
                title(['Forcing Term: run (gold context) vs run (dataset context), dimension ', num2str(d)]);
                legend('run (gold context)','run (dataset context)');
            hold off;
    end

    figure;
    for d=1:Dq
        subplot(Dq,1,d);
            hold on;
                plot(QT_gold(d,:),'r');
                plot(Q_hist_gc(d,:),'g');
                plot(Q_hist(d,:),'b');
                fprintf('Maximum Absolute Difference in Q%d = %f\n', d, max(abs(Q_hist(d,:)-QT_gold(d,:))));
                title(['Q: gold vs run (gold context) vs run (dataset context), dimension ', num2str(d)]);
                legend('gold', 'run (gold context)', 'run (dataset context)');
            hold off;
    end

    figure;
    for d=1:Dq
        subplot(Dq,1,d);
            hold on;
                plot(QdT_gold(d,:),'r');
                plot(Qd_hist_gc(d,:),'g');
                plot(Qd_hist(d,:),'b');
                title(['Qd: gold vs run (gold context) vs run (dataset context), dimension ', num2str(d)]);
                legend('gold', 'run (gold context)', 'run (dataset context)');
            hold off;
    end

    figure;
    for d=1:Dq
        subplot(Dq,1,d);
            hold on;
                plot(QddT_gold(d,:),'r');
                plot(Qdd_hist_gc(d,:),'g');
                plot(Qdd_hist(d,:),'b');
                title(['Qdd: gold vs run (gold context) vs run (dataset context), dimension ', num2str(d)]);
                legend('gold', 'run (gold context)', 'run (dataset context)');
            hold off;
    end
end

%% Plotting Unrolling Results versus Quaternion Demo Trajectories
% Plot the original Q trajectory as red curve,
% the unrolled one as green curve, and 
% the perturbed ones as blue curves 
% (stretched back to the original traj_length):
figure;
for d=1:Dq
    subplot(Dq,1,d);
    hold on;
        %plot(QT_gold(d,:), 'r');
        plot(Q_hist(d,:),'g','LineWidth',3);
        for i=1:N_demo
            perturbed_traj_length   = size(Quat_traj_demo_set{1,i}, 2);
            %plot(Quat_traj_demo_set{1,i}(d,:), 'b-.');
            plot(stretchTrajectory(Quat_traj_demo_set{1,i}(d,:), traj_length_gold), 'b-.');
        end
        title(['dmp-unrolled vs (stretched) Q demo trajectories, dimension ', num2str(d)]);
        legend('dmp-unrolled', 'demos');
    hold off;
end

% Plot the original omega trajectory as red curve,
% the unrolled one as green curve, and 
% the perturbed ones as blue curves 
% (stretched and scaled back by tau ratio):
figure;
for d=1:D
    subplot(D,1,d);
    hold on;
        %plot(omegaT_gold(d,:), 'r');
        plot(omega_hist(d,:),'g','LineWidth',3);
        for i=1:N_demo
            perturbed_traj_length   = size(Quat_traj_demo_set{2,i}, 2);
            %plot(Quat_traj_demo_set{2,i}(d,:), 'b-.');
            plot(((perturbed_traj_length-1)/(traj_length_gold-1)) * stretchTrajectory(Quat_traj_demo_set{2,i}(d,:), traj_length_gold), 'b-.');
        end
        title(['dmp-unrolled vs (stretched and scaled) omega demo trajectories, dimension ', num2str(d)]);
        legend('dmp-unrolled', 'demos');
    hold off;
end

% Plot the original omegad trajectory as red curve,
% the unrolled one as green curve, and 
% the perturbed ones as blue curves 
% (stretched and scaled back by squared tau ratio):
figure;
for d=1:D
    subplot(D,1,d);
    hold on;
        %plot(omegadT_gold(d,:), 'r');
        plot(omegad_hist(d,:),'g','LineWidth',3);
        for i=1:N_demo
            perturbed_traj_length   = size(Quat_traj_demo_set{3,i}, 2);
            %plot(Quat_traj_demo_set{3,i}(d,:), 'b-.');
            plot(((perturbed_traj_length-1)/(traj_length_gold-1))^2 * stretchTrajectory(Quat_traj_demo_set{3,i}(d,:), traj_length_gold), 'b-.');
        end
        title(['dmp-unrolled vs (stretched and scaled) omegad demo trajectories, dimension ', num2str(d)]);
        legend('dmp-unrolled', 'demos');
    hold off;
end