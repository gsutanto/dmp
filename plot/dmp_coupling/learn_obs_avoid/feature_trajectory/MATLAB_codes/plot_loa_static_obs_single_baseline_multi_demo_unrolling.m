function plot_loa_static_obs_single_baseline_multi_demo_unrolling( varargin )
    % Author: Giovanni Sutanto
    % Date  : January 05, 2016
    close   all;
    clc;
    
    in_data_path    = '../../../../../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_static/';
    
    % count number of available static obstacle settings:
    i = 1;
    while (exist(strcat(in_data_path, num2str(i)), 'dir'))
        i                           = i + 1;
    end
    num_settings                    = i - 1;
    
    if (nargin > 0)
        num_settings                = varargin{1};
    end
    
    for n = 1:num_settings
        obs_sph_center_coord        = dlmread(strcat(in_data_path, num2str(n), '/obs_sph_center_coord.txt'));
        obs_sph_radius              = dlmread(strcat(in_data_path, num2str(n), '/obs_sph_radius.txt'));
        
        unrolled_traj_wo_obs        = dlmread('unroll_tests/baseline/transform_sys_state_global_trajectory.txt');
        averaged_traj_w_obs         = dlmread(strcat(num2str(n), '/transform_sys_state_global_trajectory.txt'));
        unrolled_traj_w_obs_ideal   = dlmread(strcat('unroll_tests/', num2str(n), '/ideal/transform_sys_state_global_trajectory.txt'));
        unrolled_traj_w_obs_real    = dlmread(strcat('unroll_tests/', num2str(n), '/real/transform_sys_state_global_trajectory.txt'));

        figure;
        hold            on;
        grid            on;
        plot_sphere(obs_sph_radius, obs_sph_center_coord(1,1),      obs_sph_center_coord(2,1),      obs_sph_center_coord(3,1));
        plot_sphere(0.01,           unrolled_traj_wo_obs(end,2),    unrolled_traj_wo_obs(end,3),    unrolled_traj_wo_obs(end,4));
        po_unroll_traj_wo_obs       = plot3(unrolled_traj_wo_obs(:,2),      unrolled_traj_wo_obs(:,3),      unrolled_traj_wo_obs(:,4),      'r', 'LineWidth', 3);
        po_ave_traj_w_obs           = plot3(averaged_traj_w_obs(:,2),       averaged_traj_w_obs(:,3),       averaged_traj_w_obs(:,4),       'b', 'LineWidth', 3);
        po_unroll_traj_w_obs_ideal  = plot3(unrolled_traj_w_obs_ideal(:,2), unrolled_traj_w_obs_ideal(:,3), unrolled_traj_w_obs_ideal(:,4), 'g', 'LineWidth', 3);
        po_unroll_traj_w_obs_real   = plot3(unrolled_traj_w_obs_real(:,2),  unrolled_traj_w_obs_real(:,3),  unrolled_traj_w_obs_real(:,4),  'c', 'LineWidth', 3);
        title(strcat('Plot of Learn-Obs-Avoid Averaged vs Unrolled Trajectories for Setting #', num2str(n)));
        xlabel('x');
        ylabel('y');
        zlabel('z');
        legend([po_unroll_traj_wo_obs, po_ave_traj_w_obs, po_unroll_traj_w_obs_ideal, po_unroll_traj_w_obs_real],...
               'Cartesian DMP unrolled trajectory without coupling term (BASELINE)', 'Cartesian DMP averaged trajectory with obstacle',...
               'Cartesian DMP unrolled trajectory with coupling term computed from TRAINING features',...
               'Cartesian DMP unrolled trajectory with coupling term computed from TEST features');
        campos([1.1699   -3.3745    0.5115]);
        hold            off;
    end
end
