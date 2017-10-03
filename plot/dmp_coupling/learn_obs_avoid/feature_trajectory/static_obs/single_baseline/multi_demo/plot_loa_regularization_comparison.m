function plot_loa_regularization_comparison(  )
    % A MATLAB script to plot different regularization effect on ridge
    % regression of the learn-obstacle-avoidance coupling term fit
    % on different static obstacle settings.
    % Author: Giovanni Sutanto
    % Date  : January 21, 2016
    close all;
    clc;
    
    if (exist('regularization_effect', 'dir'))
        rmdir('regularization_effect', 's');
    end
    mkdir('regularization_effect');
    
    % count number of available static obstacle settings:
    i = 1;
    while (exist(strcat('regularization_comparison/', num2str(i)), 'dir'))
        i                   = i + 1;
    end
    num_reg_const           = i - 1;
    ColorSet                = varycolor(num_reg_const+1);
    
    Ct_target               = dlmread('regularization_comparison/1/Ct_target.txt');
    
    for i = 1:num_reg_const
        Ct_fit{i}               = dlmread(strcat('regularization_comparison/', num2str(i), '/Ct_fit.txt'));
        regularization_const{i} = dlmread(strcat('regularization_comparison/', num2str(i), '/regularization_const.txt'));
    end
        
    fig     = figure('units','normalized','outerposition',[0 0 1 1]);
    for j = 1:3
        subplot(2,2,j);
        hold on;
        grid on;
        Ct_t                = plot(Ct_target(:,j), 'Color', ColorSet(end,:));
        for i = 1:num_reg_const
            Ct_f{i}         = plot(Ct_fit{i}(:,j), 'Color', ColorSet(i,:));
            reg_legend{i}   = ['regularization const = ', num2str(regularization_const{i})];
        end
        if (j == 1)
            title('x');
        elseif (j == 2)
            title('y');
        elseif (j == 3)
            title('z');
        end
        xlabel('time');
        ylabel('Ct');
        legend([Ct_t, Ct_f{:}], 'Ct_t_a_r_g_e_t', reg_legend{:});
        hold off;
    end
    saveas(fig, 'regularization_effect/Ct_fit_for_Different_Regularization_Constant.jpg');
    
    mkdir('regularization_effect/local_trajectory_comparison/');
    for i = 1:num_reg_const
        averaged_traj_w_obs         = dlmread(strcat('regularization_comparison/', num2str(i), '/1/transform_sys_state_local_trajectory.txt'));
        unrolled_traj_w_obs_ideal   = dlmread(strcat('regularization_comparison/', num2str(i), '/unroll_tests/1/ideal/transform_sys_state_local_trajectory.txt'));
        unrolled_traj_w_obs_real    = dlmread(strcat('regularization_comparison/', num2str(i), '/unroll_tests/1/real/transform_sys_state_local_trajectory.txt'));

        fig     = figure('units','normalized','outerposition',[0 0 1 1]);
        for j = 1:3
            subplot(2,2,j);
            hold on;
            grid on;
            ave                 = plot(averaged_traj_w_obs(:,1),        averaged_traj_w_obs(:,j+1), 'Color', 'b');
            uroll_ideal         = plot(unrolled_traj_w_obs_ideal(:,1),  unrolled_traj_w_obs_ideal(:,j+1), 'Color', 'g');
            uroll_real          = plot(unrolled_traj_w_obs_real(:,1),   unrolled_traj_w_obs_real(:,j+1), 'Color', 'c');
            xlabel('time');
            if (j == 1)
                title(strcat('regularization const=',num2str(regularization_const{i})));
                ylabel('x');
            elseif (j == 2)
                ylabel('y');
            elseif (j == 3)
                ylabel('z');
            end
            legend([ave, uroll_ideal, uroll_real], 'Averaged Trajectory w/ Obs', 'Unrolled Trajectory (Ideal)', 'Unrolled Trajectory (Real)');
            hold off;
        end
        saveas(fig, strcat('regularization_effect/local_trajectory_comparison/trajectory_in_local_coordinate_setting_1_w_reg_const_', num2str(regularization_const{i}), '.jpg'));
    end
end
