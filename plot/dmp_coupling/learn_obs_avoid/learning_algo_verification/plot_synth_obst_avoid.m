close all;
clear all;
clc;

data_path                               = '../../../../data/dmp_coupling/learn_obs_avoid/learning_algo_verification/';

dmp_traj_baseline_local                 = dlmread('dmp/baseline/transform_sys_state_local_trajectory.txt');
dmp_traj_w_synth_obs_avoid_local        = dlmread('dmp/w_synthetic_obs_avoidance/transform_sys_state_local_trajectory.txt');
dmp_traj_w_ARD_learned_obs_avoid_local  = dlmread('dmp/w_ARD_learned_obs_avoidance/transform_sys_state_local_trajectory.txt');

dmp_traj_baseline_global                = dlmread('dmp/baseline/transform_sys_state_global_trajectory.txt');
dmp_traj_w_synth_obs_avoid_global       = dlmread([data_path, 'synthetic_data/1/endeff_trajs/1.txt']);
dmp_traj_w_ARD_learned_obs_avoid_global = dlmread('dmp/w_ARD_learned_obs_avoidance/transform_sys_state_global_trajectory.txt');
obs_sph_center_coord                    = dlmread([data_path, 'synthetic_data/1/obs_sph_center_coord.txt']);
obs_sph_radius                          = dlmread([data_path, 'synthetic_data/1/obs_sph_radius.txt']);

X_synth_obs_avoid           = dlmread('dmp/w_synthetic_obs_avoidance/feature_trajectory.txt');
X_target_obs_avoid          = dlmread('loa/X.txt');

diff_X_target_synth         = X_target_obs_avoid - X_synth_obs_avoid;

Ct_synth_obs_avoid          = dlmread('dmp/w_synthetic_obs_avoidance/transform_sys_ct_acc_trajectory.txt');
Ct_target_obs_avoid         = dlmread('loa/Ct_target.txt');
Ct_target_obs_avoid         = [Ct_synth_obs_avoid(:,1), Ct_target_obs_avoid];   % copy in timing information
Ct_ARD_learned_obs_avoid    = dlmread('dmp/w_ARD_learned_obs_avoidance/transform_sys_ct_acc_trajectory.txt');

diff_Ct_target_synth        = Ct_target_obs_avoid(:,2:4) - Ct_synth_obs_avoid(:,2:4);
diff_Ct_target_ARD_learned  = Ct_target_obs_avoid(:,2:4) - Ct_ARD_learned_obs_avoid(:,2:4);

disp(['------------------------------------------']);
disp(['MSE X_target_synth                   = ', num2str(mean(mean(diff_X_target_synth.^2)))]);
disp(['Max Abs(diff_X_target_synth)         = ', num2str(max(max(abs(diff_X_target_synth))))]);
disp(['Min Abs(diff_X_target_synth)         = ', num2str(min(min(abs(diff_X_target_synth))))]);
disp(['------------------------------------------']);
disp(['MSE Ct_target_synth                  = ', num2str(mean(mean(diff_Ct_target_synth.^2)))]);
disp(['Max Abs(diff_Ct_target_synth)        = ', num2str(max(max(abs(diff_Ct_target_synth))))]);
disp(['Min Abs(diff_Ct_target_synth)        = ', num2str(min(min(abs(diff_Ct_target_synth))))]);
disp(['------------------------------------------']);
disp(['MSE Ct_target_ARD_learned            = ', num2str(mean(mean(diff_Ct_target_ARD_learned.^2)))]);
disp(['Max Abs(diff_Ct_target_ARD_learned)  = ', num2str(max(max(abs(diff_Ct_target_ARD_learned))))]);
disp(['Min Abs(diff_Ct_target_ARD_learned)  = ', num2str(min(min(abs(diff_Ct_target_ARD_learned))))]);
disp(['------------------------------------------']);

D                           = 3;

figure;
axis equal;
grid on;
hold on;
    plot3(dmp_traj_baseline_global(:,2),dmp_traj_baseline_global(:,3),dmp_traj_baseline_global(:,4),'b');
    plot3(dmp_traj_w_synth_obs_avoid_global(:,2),dmp_traj_w_synth_obs_avoid_global(:,3),dmp_traj_w_synth_obs_avoid_global(:,4),'g');
    plot3(dmp_traj_w_ARD_learned_obs_avoid_global(:,2),dmp_traj_w_ARD_learned_obs_avoid_global(:,3),dmp_traj_w_ARD_learned_obs_avoid_global(:,4),'r');
    plot_sphere(obs_sph_radius, obs_sph_center_coord(1,1), obs_sph_center_coord(2,1), obs_sph_center_coord(3,1));
    xlabel('x');
    ylabel('y');
    zlabel('z');
    legend('baseline', 'synth', 'ARD-learned');
    title('Obstacle Avoidance Behavior in Global Coordinate System');
hold off;

figure;
axis equal;
grid on;
hold on;
    plot3(Ct_synth_obs_avoid(:,2),Ct_synth_obs_avoid(:,3),Ct_synth_obs_avoid(:,4),'b');
    plot3(Ct_target_obs_avoid(:,2),Ct_target_obs_avoid(:,3),Ct_target_obs_avoid(:,4),'g');
    plot3(Ct_ARD_learned_obs_avoid(:,2),Ct_ARD_learned_obs_avoid(:,3),Ct_ARD_learned_obs_avoid(:,4),'r');
    xlabel('x');
    ylabel('y');
    zlabel('z');
    legend('synth', 'target', 'ARD-learned');
    title('ct\_synth vs ct\_target vs ct\_ard\_learned (3D)');
hold off;

figure;
axis equal;
grid on;
for d=1:D
    subplot(D,1,d);
    hold on;
        plot(Ct_synth_obs_avoid(:,d+1),'b');
        plot(Ct_target_obs_avoid(:,d+1),'g');
        plot(Ct_ARD_learned_obs_avoid(:,d+1),'r');
        ystring     = ['dim #',num2str(d)];
        ylabel(ystring);
        legend('synth', 'target', 'ARD-learned');
    hold off;
    if (d==1)
        title('ct\_synth vs ct\_target vs ct\_ard\_learned (per dimension)');
    end
end