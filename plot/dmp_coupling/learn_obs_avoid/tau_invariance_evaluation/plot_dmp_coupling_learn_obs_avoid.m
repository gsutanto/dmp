function plot_dmp_coupling_learn_obs_avoid(obstacle_position_selection)
    % Author: Giovanni Sutanto
    % Date  : November 10, 2015
    close   all;
    
    sample_cart_dmp_wo_obs_out                      = dlmread('wo_obs/transform_sys_state_global_trajectory.txt');
    sample_cart_dmp_w_obs_out_dmp_unroll            = dlmread('w_obs_dmp_unroll/transform_sys_state_global_trajectory.txt');
    sample_cart_dmp_w_obs_out_ideal                 = dlmread('w_obs_ideal/transform_sys_state_global_trajectory.txt');
    sample_cart_dmp_w_obs_out_real                  = dlmread('w_obs_real/transform_sys_state_global_trajectory.txt');
    
    figure;
    hold            on;
    grid            on;
%     px              = quiver3(0,0,0,1,0,0,'r');
%     py              = quiver3(0,0,0,0,1,0,'g');
%     pz              = quiver3(0,0,0,0,0,1,'b');
    po_cart_dmp_wo_obs              = plot3(sample_cart_dmp_wo_obs_out(:,2)', sample_cart_dmp_wo_obs_out(:,3)', sample_cart_dmp_wo_obs_out(:,4)','r', 'LineWidth', 3);
    po_cart_dmp_w_obs_dmp_unroll    = plot3(sample_cart_dmp_w_obs_out_dmp_unroll(:,2)', sample_cart_dmp_w_obs_out_dmp_unroll(:,3)', sample_cart_dmp_w_obs_out_dmp_unroll(:,4)','b', 'LineWidth', 3);
    po_cart_dmp_w_obs_ideal         = plot3(sample_cart_dmp_w_obs_out_ideal(:,2)', sample_cart_dmp_w_obs_out_ideal(:,3)', sample_cart_dmp_w_obs_out_ideal(:,4)','g', 'LineWidth', 3);
    po_cart_dmp_w_obs_real          = plot3(sample_cart_dmp_w_obs_out_real(:,2)', sample_cart_dmp_w_obs_out_real(:,3)', sample_cart_dmp_w_obs_out_real(:,4)','c', 'LineWidth', 3);
    if (obstacle_position_selection == 0)
        plot_sphere(0.05, 0.553216, -0.016969, 0.225269);
    elseif (obstacle_position_selection == 1)
        plot_sphere(0.05, 0.299291, 0.491815, 0.266131);
    end
    title('Plot of Cartesian DMP Performance');
%     legend([px, py, pz, po_cart_dmp_w_obs_dmp_unroll, po_cart_dmp_wo_obs, po_cart_dmp_w_obs_ideal, po_cart_dmp_w_obs_real],...
%            'global x-axis','global y-axis','global z-axis',...
%            'cartesian DMP unrolled trajectory with obstacle',...
%            'cartesian DMP unrolled trajectory without coupling term','cartesian DMP unrolled trajectory with coupling term computed from TRAINING features',...
%            'cartesian DMP unrolled trajectory with coupling term computed from TEST features');
    legend([po_cart_dmp_wo_obs, po_cart_dmp_w_obs_dmp_unroll, po_cart_dmp_w_obs_ideal, po_cart_dmp_w_obs_real],...
           'Cartesian DMP unrolled trajectory without coupling term (BASELINE)', 'Cartesian DMP averaged trajectory with obstacle',...
           'Cartesian DMP unrolled trajectory with coupling term computed from TRAINING features',...
           'Cartesian DMP unrolled trajectory with coupling term computed from TEST features');
    campos([1.1699   -3.3745    0.5115]);
    hold            off;
end
