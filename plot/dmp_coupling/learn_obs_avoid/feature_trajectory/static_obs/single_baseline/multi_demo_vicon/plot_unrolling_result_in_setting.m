function [norm_end_eff_xddyddzdd_demo] = plot_unrolling_result_in_setting( varargin )
    % Example command calling:
    % plot_unrolling_result_in_setting('learned_weights/2/',118,1);
    
    unrolling_subdir            = varargin{1};
    setting_num                 = varargin{2};
    demo_in_setting_num         = varargin{3};
    if (nargin > 3)
        is_plot_unrolling       = varargin{4};
    else
        is_plot_unrolling       = 1;
    end

    close all;
    clc;

    addpath('../../../../../../../matlab/dmp_coupling/learn_obs_avoid/utilities/');
    
    vicon_marker_radius         = 15.0/2000.0;  % in meter
    hand_marker_radius          = 50/1000.0;    % in meter

    data_dir                    = '../../../../../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_vicon_static/';
    unrolling_dir               = [unrolling_subdir, 'unroll_tests/', num2str(setting_num), '/', num2str(demo_in_setting_num), '/'];
    
    obs_markers_global_xyz      = dlmread([data_dir, num2str(setting_num), '/obs_markers_global_coord.txt']);
    
    end_eff_xyz_demo_traj       = dlmread([data_dir, num2str(setting_num), '/endeff_trajs/', num2str(demo_in_setting_num), '.txt']);
    end_eff_xyz_demo            = end_eff_xyz_demo_traj(:,2:4);
    end_eff_xdydzd_demo         = end_eff_xyz_demo_traj(:,5:7);
    end_eff_xddyddzdd_demo      = end_eff_xyz_demo_traj(:,8:10);
    dt                          = end_eff_xyz_demo_traj(2,1)-end_eff_xyz_demo_traj(1,1);
    tau                         = (size(end_eff_xyz_demo_traj,1)-1)*dt;
    
    norm_end_eff_xddyddzdd_demo = zeros(size(end_eff_xyz_demo_traj,1),1);
    
    for i=1:size(end_eff_xyz_demo_traj,1)
        norm_end_eff_xddyddzdd_demo(i,1)    = tau * tau * norm(end_eff_xddyddzdd_demo(i,:));
    end
    diff_norm_end_eff_xddyddzdd_demo        = diff(norm_end_eff_xddyddzdd_demo);
    disp(['max tau*tau*ydd = ', num2str(max(norm_end_eff_xddyddzdd_demo))]);
    disp(['max diff tau*tau*ydd = ', num2str(max(diff_norm_end_eff_xddyddzdd_demo))]);
    
    end_eff_xyz_unroll_traj     = dlmread([unrolling_dir, 'transform_sys_state_global_trajectory.txt']);
    end_eff_xyz_unroll          = end_eff_xyz_unroll_traj(:,2:4);
    
    figure;
    axis equal;
    hold on;
        plot3(end_eff_xyz_demo(:,1),...
              end_eff_xyz_demo(:,2),...
              end_eff_xyz_demo(:,3),'b');
        if (is_plot_unrolling)
            plot3(end_eff_xyz_unroll(:,1),...
                  end_eff_xyz_unroll(:,2),...
                  end_eff_xyz_unroll(:,3),'g');
        end
        plot_sphere(hand_marker_radius,...
                    end_eff_xyz_demo(end,1),...
                    end_eff_xyz_demo(end,2),...
                    end_eff_xyz_demo(end,3));
        for j=1:size(obs_markers_global_xyz,1)
            plot_sphere(vicon_marker_radius,...
                        obs_markers_global_xyz(j,1),...
                        obs_markers_global_xyz(j,2),...
                        obs_markers_global_xyz(j,3))
        end
        if (is_plot_unrolling)
            legend('demo', 'unroll');
        else
            legend('demo');
        end
        xlabel('x');
        ylabel('y');
        zlabel('z');
    hold off;
end