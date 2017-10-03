function plot_cart_coord_dmp_single_traj_training()
    % Author: Giovanni Sutanto
    % Date  : June 30, 2015
    close   all;
    
    sample_cart_dmp_in   = dlmread('../../../data/cart_dmp/cart_coord_dmp/single_traj_training/sample_traj_3D_1.txt');
    sample_cart_dmp_goal = dlmread('goal_state_global_trajectory.txt');
    sample_cart_dmp_out  = dlmread('transform_sys_state_global_trajectory.txt');
    
    figure;
    hold            on;
    grid            on;
    plot(sample_cart_dmp_goal(:,1), sample_cart_dmp_goal(:,2),...
         sample_cart_dmp_goal(:,1), sample_cart_dmp_goal(:,3),...
         sample_cart_dmp_goal(:,1), sample_cart_dmp_goal(:,4));
    title('Plot of Goal Trajectory versus Time');
    legend('X-Goal Trajectory versus Time', 'Y-Goal Trajectory versus Time', 'Z-Goal Trajectory versus Time');
    hold            off;
    
    figure;
    hold            on;
    plot3(sample_cart_dmp_in(:,2), sample_cart_dmp_in(:,3), sample_cart_dmp_in(:,4),'c');
    plot3(sample_cart_dmp_out(:,2), sample_cart_dmp_out(:,3), sample_cart_dmp_out(:,4),'m');
    plot3(sample_cart_dmp_goal(:,2), sample_cart_dmp_goal(:,3), sample_cart_dmp_goal(:,4),'y');
    quiver3(0,0,0,1,0,0,'r');
    quiver3(0,0,0,0,1,0,'g');
    quiver3(0,0,0,0,0,1,'b');
    title('Plot of Cartesian DMP Performance');
    legend('cartesian dmp: training trajectory','cartesian dmp: reproduced trajectory','cartesian dmp: goal-changing trajectory','global x-axis','global y-axis','global z-axis');
    hold            off;
    
    sample_cart_dmp_f_target    = dlmread('f_target.txt');
    sample_cart_dmp_f_fit       = dlmread('f_fit.txt');
    
    figure;
    title('Plot of Demonstrated Trajectory vs Unrolled Trajectory');
    for j = 1:3
        subplot(2,2,j);
        hold on;
        grid on;
        traj_demo   = plot(sample_cart_dmp_in(:,1), sample_cart_dmp_in(:,(1+j)), 'c');
        traj_unroll = plot(sample_cart_dmp_out(:,1), sample_cart_dmp_out(:,(1+j)), 'm');
        if (j == 1)
            title('x');
        elseif (j == 2)
            title('y');
        elseif (j == 3)
            title('z');
        end
        xlabel('time step');
        ylabel('');
        legend([traj_demo, traj_unroll], 'demo trajectory', 'unrolled trajectory');
        hold off;
    end
    
    figure;
    title('Plot of Forcing Term Target vs Forcing Term Fit Trajectory');
    for j = 1:3
        subplot(2,2,j);
        hold on;
        grid on;
        f_target    = plot(sample_cart_dmp_in(:,1), sample_cart_dmp_f_target(j,:), 'c');
        f_fit       = plot(sample_cart_dmp_in(:,1), sample_cart_dmp_f_fit(j,:), 'm');
        %err     = sample_cart_dmp_f_target(j,:) - sample_cart_dmp_f_fit(j,:);
        %mse     = err.' * err
        if (j == 1)
            title('x');
        elseif (j == 2)
            title('y');
        elseif (j == 3)
            title('z');
        end
        xlabel('time step');
        ylabel('f magnitude');
        legend([f_target, f_fit], 'f_t_a_r_g_e_t', 'f_f_i_t');
        hold off;
    end
end
