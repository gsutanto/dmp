% Quaternion Dynamic Movement Primitive (DMP) Simulation
% -Learning from Single Quaternion Trajectory Demonstration-
% 
% Author : Giovanni Sutanto
% Date   : October 19, 2016

clear all;
close all;
clc;

addpath('../../../utilities/');
addpath('../');

D   = 3;
Dq  = 4;

% Load Human Pose Quaternion Trajectory Demonstration:
cart_Q_Qd_Qdd_traj_demo         = dlmread('../../../../data/cart_dmp/cart_dmp_wiggling/sample_quat_traj_recorded_demo_wiggling.txt');
cart_Q_omega_omegad_traj_demo   = dlmread('../../../../data/cart_dmp/cart_dmp_wiggling/sample_quat_ABGomega_traj_recorded_demo_wiggling.txt');

time        = cart_Q_Qd_Qdd_traj_demo(:,1)';
QT          = cart_Q_Qd_Qdd_traj_demo(:,2:5)';
QdT         = cart_Q_Qd_Qdd_traj_demo(:,6:9)';
QddT        = cart_Q_Qd_Qdd_traj_demo(:,10:13)';
omegaT      = cart_Q_omega_omegad_traj_demo(:,6:8)';
omegadT     = cart_Q_omega_omegad_traj_demo(:,9:11)';

tau         = time(1,end) - time(1,1);
traj_length = size(QT,2);
dt          = tau/(traj_length-1);

% Goal Orientation Quaternion:
QG      = QT(:,end);
QG      = QG/norm(QG);

% Initial Orientation Quaternion:
Q0      = QT(:,1);
Q0      = Q0/norm(Q0);

N_tau_multiplier    = 3;

Q_hist                  = zeros(Dq, N_tau_multiplier*traj_length, N_tau_multiplier);
Qd_hist                 = zeros(Dq, N_tau_multiplier*traj_length, N_tau_multiplier);
Qdd_hist                = zeros(Dq, N_tau_multiplier*traj_length, N_tau_multiplier);
F_run                   = zeros(D, N_tau_multiplier*traj_length, N_tau_multiplier);

global  dcps;

n_rfs   = 25;
c_order = 1;
w     	= zeros(n_rfs,3);
ID      = 1;

dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
dcp_quaternion('reset_state', ID, Q0);
dcp_quaternion('set_goal', ID, QG, 1);

[w, F_target, F_fit, dG]    = dcp_quaternion('batch_fit', ID, tau, dt, QT, omegaT, omegadT);

% Testing Tau-Invariance:
for tau_multiplier = 1:N_tau_multiplier
    dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
    dcp_quaternion('reset_state', ID, Q0);
    dcp_quaternion('set_goal', ID, QG, 1);
    dcps(1).w   = w;

    for i=1:(traj_length*tau_multiplier)
        [Q, Qd, Qdd, ~, ~, f] = dcp_quaternion('run', ID, (tau*tau_multiplier), dt);

        Q_hist(:,i,tau_multiplier)  = Q;
        Qd_hist(:,i,tau_multiplier) = Qd;
        Qdd_hist(:,i,tau_multiplier)= Qdd;

        F_run(:,i,tau_multiplier)   = f;
    end
end

figure;
for d=1:D
    subplot(D,1,d);
        hold on;
            plot(F_target(d,:),'r');
            plot(F_fit(d,:),'g');
            plot(F_run(d,1:traj_length,1),'b');
            title(['Forcing Term: target vs fit vs unroll, dimension ', num2str(d)]);
            legend('target','fit','unroll');
        hold off;
end

figure;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(QT(d,:),'r');
            plot(Q_hist(d,1:traj_length,1),'b');
%             ylim([-1 1])
            fprintf('Maximum Absolute Difference in Q%d = %f\n', d, max(abs(Q_hist(d,1:traj_length,1)-QT(d,:))));
            title(['Q: target/demo vs unroll, dimension ', num2str(d)]);
            legend('target/demo','unroll');
        hold off;
end

figure;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(QdT(d,:),'r');
            plot(Qd_hist(d,1:traj_length,1),'b');
            title(['Qd: target/demo vs unroll, dimension ', num2str(d)]);
            legend('target/demo','unroll');
        hold off;
end

figure;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(QddT(d,:),'r');
            plot(Qdd_hist(d,1:traj_length,1),'b');
            title(['Qdd: target/demo vs unroll, dimension ', num2str(d)]);
            legend('target/demo','unroll');
        hold off;
end

% Interpolation for Evaluating the Tau-Invariance Property:
QI      = zeros(4, N_tau_multiplier*traj_length, N_tau_multiplier);
QdI     = zeros(4, N_tau_multiplier*traj_length, N_tau_multiplier);
QddI    = zeros(4, N_tau_multiplier*traj_length, N_tau_multiplier);
FrunI   = zeros(3, N_tau_multiplier*traj_length, N_tau_multiplier);
for tm = 1:N_tau_multiplier
    x               = 1:(tm*traj_length);
    xI              = 1:((tm*traj_length-1)/(N_tau_multiplier*traj_length-1)):(tm*traj_length);
    for d=1:Dq
        QI(d,:,tm)  = interp1(x,Q_hist(d,1:(tm*traj_length),tm),xI,'spline');
        QdI(d,:,tm) = interp1(x,Qd_hist(d,1:(tm*traj_length),tm),xI,'spline');
        QddI(d,:,tm)= interp1(x,Qdd_hist(d,1:(tm*traj_length),tm),xI,'spline');
        if (d <= 3)
            FrunI(d,:,tm)   = interp1(x,F_run(d,1:(tm*traj_length),tm),xI,'spline');
        end
    end
end
figure;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(QI(d,:,1),'r-.');
            plot(QI(d,:,2),'g-');
            plot(QI(d,:,3),'b:');
            title(['Q: unrolling with various tau (Tau-Invariance Test), dimension ', num2str(d)]);
            legend('1Xtau\_demo','2Xtau\_demo','3Xtau\_demo');
        hold off;
end
figure;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(QdI(d,:,1),'r-.');
            plot(QdI(d,:,2),'g-');
            plot(QdI(d,:,3),'b:');
            title(['Qd: unrolling with various tau (Tau-Invariance Test), dimension ', num2str(d)]);
            legend('1Xtau\_demo','2Xtau\_demo','3Xtau\_demo');
        hold off;
end
figure;
for d=1:Dq
    subplot(Dq,1,d);
        hold on;
            plot(QddI(d,:,1),'r-.');
            plot(QddI(d,:,2),'g-');
            plot(QddI(d,:,3),'b:');
            title(['Qdd: unrolling with various tau (Tau-Invariance Test), dimension ', num2str(d)]);
            legend('1Xtau\_demo','2Xtau\_demo','3Xtau\_demo');
        hold off;
end
figure;
for d=1:D
    subplot(D,1,d);
        hold on;
            plot(FrunI(d,:,1),'r-.');
            plot(FrunI(d,:,2),'g-');
            plot(FrunI(d,:,3),'b:');
            title(['Forcing Term Run: unrolling with various tau (Tau-Invariance Test), dimension ', num2str(d)]);
            legend('1Xtau\_demo','2Xtau\_demo','3Xtau\_demo');
        hold off;
end