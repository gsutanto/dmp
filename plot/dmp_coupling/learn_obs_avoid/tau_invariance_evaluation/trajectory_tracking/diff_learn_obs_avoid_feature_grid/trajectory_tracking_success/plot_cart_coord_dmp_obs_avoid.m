function plot_cart_coord_dmp_obs_avoid()
    % Author: Giovanni Sutanto
    % Date  : November 10, 2015
    close   all;
    
    sample_cart_coord_dmp_multi_traj_training_wo_obs_out  = csvread('sample_cart_coord_dmp_multi_traj_training_wo_obs_out.txt');
    sample_cart_coord_dmp_multi_traj_training_w_obs_out   = csvread('sample_cart_coord_dmp_multi_traj_training_w_obs_out.txt');
    sample_cart_coord_dmp_wo_obs_out                      = csvread('sample_cart_coord_dmp_wo_obs_out.txt');
    sample_cart_coord_dmp_w_obs_out_ideal                 = csvread('sample_cart_coord_dmp_w_obs_out_ideal.txt');
    sample_cart_coord_dmp_w_obs_out_real                  = csvread('sample_cart_coord_dmp_w_obs_out_real.txt');
    
    figure;
    hold            on;
    po_multi_traj_training_wo_obs   = plot3(sample_cart_coord_dmp_multi_traj_training_wo_obs_out(:,2)', sample_cart_coord_dmp_multi_traj_training_wo_obs_out(:,3)', sample_cart_coord_dmp_multi_traj_training_wo_obs_out(:,4)','mx');
    po_multi_traj_training_w_obs    = plot3(sample_cart_coord_dmp_multi_traj_training_w_obs_out(:,2)', sample_cart_coord_dmp_multi_traj_training_w_obs_out(:,3)', sample_cart_coord_dmp_multi_traj_training_w_obs_out(:,4)','bx');
    po_rbt_unroll_wo_obs            = plot3(sample_cart_coord_dmp_wo_obs_out(:,2)', sample_cart_coord_dmp_wo_obs_out(:,3)', sample_cart_coord_dmp_wo_obs_out(:,4)','r+');
    po_rbt_unroll_w_obs_ideal       = plot3(sample_cart_coord_dmp_w_obs_out_ideal(:,2)', sample_cart_coord_dmp_w_obs_out_ideal(:,3)', sample_cart_coord_dmp_w_obs_out_ideal(:,4)','g+');
    po_rbt_unroll_w_obs_real        = plot3(sample_cart_coord_dmp_w_obs_out_real(:,2)', sample_cart_coord_dmp_w_obs_out_real(:,3)', sample_cart_coord_dmp_w_obs_out_real(:,4)','c+');
    plot_sphere(0.05, 0.553216, -0.016969, 0.225269);
    title('Plot of Cartesian DMP Performance');
    legend([po_multi_traj_training_wo_obs, po_multi_traj_training_w_obs, po_rbt_unroll_wo_obs, po_rbt_unroll_w_obs_ideal, po_rbt_unroll_w_obs_real],...
           'cartesian dmp w/o obs: reproduced trajectory','cartesian dmp w/ obs: reproduced trajectory',...
           'cartesian dmp w/o obs: reproduced trajectory by robot','cartesian dmp w/ obs: reproduced trajectory by robot (ideal)',...
           'cartesian dmp w/ obs: reproduced trajectory by robot (real)');
    campos([1.5198   -3.1373    0.6603]);
    hold            off;
end
