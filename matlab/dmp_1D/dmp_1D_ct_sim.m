% Dynamic Movement Primitive (DMP) Simulation
% -After Learning a Baseline DMP Primitive, 
%  Simulate the Dummy Coupling Term (Ct) Signals Generation 
%  and the Resulting Ct-Induced Trajectories, then Finally
%  Simulate the Coupling Term (Ct) Re-Extraction from 
%  the Ct-Induced Trajectories-
% 
% Author : Giovanni Sutanto
% Date   : January 2017

clear all;
close all;
clc;

% Seed the randomness:
rng(1234);

addpath('../utilities/');
addpath('../dmp_general/');

%% DMP 1D Parameter Setup

global          dcps;

ID              = 1;
n_rfs           = 25;
c_order         = 1;

%% Data Loading

traj_1D_demo    = dlmread('../../data/dmp_1D/sample_traj_1.txt');

time            = traj_1D_demo(:,1);
tau             = time(end,1) - time(1,1);
dt              = time(2,1)   - time(1,1);
T               = traj_1D_demo(:,2);
Td              = traj_1D_demo(:,3);
Tdd             = traj_1D_demo(:,4);

start           = T(1,1);
goal            = T(end,1);

taus            = [tau];
dts             = [dt];
Ts              = {T};
Tds             = {Td};
Tdds            = {Tdd};

T0_demo_baseline= T(1);
TG_demo_baseline= T(end);

%% Learning Baseline DMP 1D Primitive
    
dcp_franzi('init', ID, n_rfs, 'dcp_1D_with_2nd_order_canonical_system', c_order);
[w, Ft, Ff, c, D, G, X, V, PSI] = dcp_franzi('batch_fit_multi', ID, taus, dts, Ts, Tds, Tdds);
[ Y, Yd, Ydd, F ]   = unrollDMP1D( w, n_rfs, 1, start, goal, dt, tau );

%% Display Baseline DMP 1D Primitive Unrolling

figure;
hold        on;
    grid    on;
    plot(time, F);
    title('Unrolled Forcing Term');
    legend('F');
hold        off;

figure;
hold        on;
    grid    on;
    plot(time, T);
    plot(time, Y);
    title('Position Trajectory');
    legend('T', 'Y');
hold        off;

clear       T Td Tdd

%% Generation of Dummy Coupled Demo Trajectories (with Non-Zero Initial Velocity and Acceleration)

fprintf('Generating (Coupled) Demo Trajectory with Non-Zero Initial Velocity and Acceleration\n');

N_demo              = 5;
Ct_const_multiplier = 50.0;
tau_demo_baseline   = tau;

T_Td_Tdd_gold_traj_demo_set = cell(N_demo, 3);
Ft_gold_traj_demo_set       = cell(N_demo, 1);
Ct_gold_traj_demo_set       = cell(N_demo, 1);

for d_idx = 1:N_demo
    fprintf('Generating (Coupled) Demo Trajectory # %d/%d\n', ...
            d_idx, N_demo);
    
    % movement duration (stretching)
    new_tau         = normrnd(tau_demo_baseline, 0.2*tau_demo_baseline);
    if (new_tau < (0.5 * tau_demo_baseline))    % some lower bound clipping
        new_tau     = 0.5 * tau_demo_baseline;
    elseif (new_tau > (1.5 * tau_demo_baseline))% some upper bound clipping
        new_tau     = 1.5 * tau_demo_baseline;
    end
    new_traj_length = round(new_tau / dt) + 1;
    new_tau         = dt * (new_traj_length - 1);
    
    T               = zeros(new_traj_length, 1);
    Td              = zeros(new_traj_length, 1);
    Tdd             = zeros(new_traj_length, 1);
    FtT             = zeros(new_traj_length, 1);
    CtT             = zeros(new_traj_length, 1);
    Ct              = 0;
    
    new_Td0         = normrnd(0, 5.0);  % randomized initial velocity
    new_Tdd0        = normrnd(0, 1.0);  % randomized initial acceleration

    dcp_franzi('init', ID, n_rfs, num2str(ID), c_order);
    dcp_franzi('reset_state', ID, T0_demo_baseline, new_Td0, new_Tdd0, new_tau);
    dcp_franzi('set_goal', ID, TG_demo_baseline, 1);
    dcps(ID).w      = w;

    for i=1:new_traj_length
        t           = (i-1) * dt;
        
        Ct          = sin(2*pi*t/new_tau);
        % add some Gaussian noise:
        if ((i ~= 1) && (i ~= new_traj_length))
            Ct      = Ct + normrnd(0, 0.2);
        end
        Ct          = Ct_const_multiplier * Ct;
        
        [y,yd,ydd,f]= dcp_franzi('run', ID, new_tau, dt, Ct);
        
        T(i,1)      = y;
        Td(i,1)     = yd;
        Tdd(i,1)    = ydd;
        FtT(i,1)    = f;
        CtT(i,1)    = Ct;
    end
    
    T_Td_Tdd_gold_traj_demo_set{d_idx, 1}   = T;
    T_Td_Tdd_gold_traj_demo_set{d_idx, 2}   = Td;
    T_Td_Tdd_gold_traj_demo_set{d_idx, 3}   = Tdd;
    
    Ft_gold_traj_demo_set{d_idx, 1}         = FtT;
    Ct_gold_traj_demo_set{d_idx, 1}         = CtT;
end

% Plot the per-demonstration baseline versus coupled trajectories:
for d_idx=1:N_demo
    figure;
    hold on;
        plot(stretchTrajectory(Y.', length(T_Td_Tdd_gold_traj_demo_set{d_idx, 1})).', 'b');
        plot(T_Td_Tdd_gold_traj_demo_set{d_idx, 1}, 'g');
        title(['baseline vs coupled position, trajectory # ', num2str(d_idx)]);
        legend('baseline', 'coupled');
    hold off;
    
    figure;
    hold on;
        plot(stretchTrajectory(Yd.', length(T_Td_Tdd_gold_traj_demo_set{d_idx, 2})).', 'b');
        plot(T_Td_Tdd_gold_traj_demo_set{d_idx, 2}, 'g');
        title(['baseline vs coupled velocity, trajectory # ', num2str(d_idx)]);
        legend('baseline', 'coupled');
    hold off;
    
    figure;
    hold on;
        plot(stretchTrajectory(Ydd.', length(T_Td_Tdd_gold_traj_demo_set{d_idx, 3})).', 'b');
        plot(T_Td_Tdd_gold_traj_demo_set{d_idx, 3}, 'g');
        title(['baseline vs coupled acceleration, trajectory # ', num2str(d_idx)]);
        legend('baseline', 'coupled');
    hold off;
end

%% Extract the Coupling Term Trajectories from the Demonstrations (with Non-Zero Initial Velocity and Acceleration)

Ft_extracted_traj_demo_set          = cell(N_demo, 1);
Ct_extracted_traj_demo_set          = cell(N_demo, 1);

for d_idx = 1:N_demo
    [ CtT_ext, FtT_ext ] = computeDMPCtTarget(  T_Td_Tdd_gold_traj_demo_set{d_idx, 1},...
                                                T_Td_Tdd_gold_traj_demo_set{d_idx, 2},...
                                                T_Td_Tdd_gold_traj_demo_set{d_idx, 3},...
                                                w, n_rfs,...
                                                T0_demo_baseline, TG_demo_baseline,...
                                                dt, c_order );
    
    Ft_extracted_traj_demo_set{d_idx, 1}    = FtT_ext;
    Ct_extracted_traj_demo_set{d_idx, 1}    = CtT_ext;
end

%% Evaluation: Measure Difference between Extracted Ct and Gold/Ground Truth Ct (with Non-Zero Initial Velocity and Acceleration)

Ct_gold     = cell2mat(Ct_gold_traj_demo_set);
Ct_extracted= cell2mat(Ct_extracted_traj_demo_set);
rmse_Ct     = sqrt(mean(mean((Ct_gold-Ct_extracted).^2)));
fprintf('RMSE Ct = %f\n', rmse_Ct);

Ft_gold     = cell2mat(Ft_gold_traj_demo_set);
Ft_extracted= cell2mat(Ft_extracted_traj_demo_set);
rmse_Ft     = sqrt(mean(mean((Ft_gold-Ft_extracted).^2)));
fprintf('RMSE Ft = %f\n', rmse_Ft);

%% Plotting (with Non-Zero Initial Velocity and Acceleration)
% Plot the all Ct ground-truth trajectories as blue curves vs
% extracted Ct trajectories as green curves
% (stretched to have equal lengths):
figure;
hold on;
    for d_idx=1:N_demo
        plot(Ct_gold_traj_demo_set{d_idx,1}, 'b-.');
        plot(Ct_extracted_traj_demo_set{d_idx,1}, 'g*');
    end
    title(['ground-truth vs extracted Coupling Term trajectories']);
    legend('ground-truth', 'extracted');
hold off;

% Plot the per-demonstration Ct ground-truth trajectory as blue curve vs
% extracted Ct trajectory as green curve:
for d_idx=1:N_demo
    figure;
    hold on;
        plot(Ct_gold_traj_demo_set{d_idx,1}, 'b-.');
        plot(Ct_extracted_traj_demo_set{d_idx,1}, 'g*');
        title(['ground-truth vs extracted Coupling Term trajectory # ', num2str(d_idx)]);
        legend('ground-truth', 'extracted');
    hold off;
end