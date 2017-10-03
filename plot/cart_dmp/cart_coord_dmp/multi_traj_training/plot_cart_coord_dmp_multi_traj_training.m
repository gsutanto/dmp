function plot_cart_coord_dmp_multi_traj_training( is_plot_training_trajectories )
    % Author: Giovanni Sutanto
    % Date  : June 30, 2015
    close   all;
    
    sample_cart_coord_dmp_multi_traj_training_wo_obs_in         = cell(0, 0);
    baseline_data_dir_path                                      = './../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/baseline/endeff_trajs/';
    files   = dir(strcat(baseline_data_dir_path,'*.txt'));
    i       = 0;
    for file = files'
        i                                                       = i + 1;
        sample_cart_coord_dmp_multi_traj_training_wo_obs_in{i}  = dlmread(strcat(baseline_data_dir_path,file.name));
    end
    sample_cart_coord_dmp_multi_traj_training_wo_obs_out        = dlmread('wo_obs/transform_sys_state_global_trajectory.txt');
    
    sample_cart_coord_dmp_multi_traj_training_w_obs_in          = cell(0, 0);
    w_obs_data_dir_path                                         = './../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/1/endeff_trajs/';
    files   = dir(strcat(w_obs_data_dir_path,'*.txt'));
    i       = 0;
    for file = files'
        i                                                       = i + 1;
        sample_cart_coord_dmp_multi_traj_training_w_obs_in{i}   = dlmread(strcat(w_obs_data_dir_path,file.name));
    end
    sample_cart_coord_dmp_multi_traj_training_w_obs_out         = dlmread('w_obs/transform_sys_state_global_trajectory.txt');
    
    figure;
    hold            on;
    grid on;
    px  = quiver3(0,0,0,1,0,0,'r');
    py  = quiver3(0,0,0,0,1,0,'g');
    pz  = quiver3(0,0,0,0,0,1,'b');
    if (is_plot_training_trajectories == 1)
        for i = 1:length(sample_cart_coord_dmp_multi_traj_training_wo_obs_in)
            pi_wo_obs   = plot3(sample_cart_coord_dmp_multi_traj_training_wo_obs_in{i}(:,2)', sample_cart_coord_dmp_multi_traj_training_wo_obs_in{i}(:,3)', sample_cart_coord_dmp_multi_traj_training_wo_obs_in{i}(:,4)','c:');
        end
    end
    po_wo_obs       = plot3(sample_cart_coord_dmp_multi_traj_training_wo_obs_out(:,2)', sample_cart_coord_dmp_multi_traj_training_wo_obs_out(:,3)', sample_cart_coord_dmp_multi_traj_training_wo_obs_out(:,4)','mx');
    if (is_plot_training_trajectories == 1)
        for i = 1:length(sample_cart_coord_dmp_multi_traj_training_w_obs_in)
            pi_w_obs    = plot3(sample_cart_coord_dmp_multi_traj_training_w_obs_in{i}(:,2)', sample_cart_coord_dmp_multi_traj_training_w_obs_in{i}(:,3)', sample_cart_coord_dmp_multi_traj_training_w_obs_in{i}(:,4)','b:');
        end
    end
    po_w_obs        = plot3(sample_cart_coord_dmp_multi_traj_training_w_obs_out(:,2)', sample_cart_coord_dmp_multi_traj_training_w_obs_out(:,3)', sample_cart_coord_dmp_multi_traj_training_w_obs_out(:,4)','rx');
    plot_sphere(0.05, 0.553216, -0.016969, 0.225269);
    title('Plot of Cartesian DMP Performance');
    if (is_plot_training_trajectories == 1)
        legend([px, py, pz, pi_wo_obs, po_wo_obs, pi_w_obs, po_w_obs], 'global x-axis','global y-axis','global z-axis',...
               'cartesian dmp w/o obs: training trajectories','cartesian dmp w/o obs: reproduced trajectory',...
               'cartesian dmp w/ obs: training trajectories','cartesian dmp w/ obs: reproduced trajectory');
    else
        legend([px, py, pz, po_wo_obs, po_w_obs], 'global x-axis','global y-axis','global z-axis',...
               'cartesian dmp w/o obs: reproduced trajectory','cartesian dmp w/ obs: reproduced trajectory');
    end
    hold            off;
end
