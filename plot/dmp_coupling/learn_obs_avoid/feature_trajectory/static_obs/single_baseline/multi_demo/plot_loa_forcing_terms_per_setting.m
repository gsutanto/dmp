function plot_loa_forcing_terms_per_setting(  )
    % A MATLAB script to plot forcing term (w/ obstacle vs baseline) 
    % on different static obstacle settings.
    % Author: Giovanni Sutanto
    % Date  : January 20, 2016
    close all;
    clc;
    
    if (exist('forcing_term_vs_coupling_term', 'dir'))
        rmdir('forcing_term_vs_coupling_term', 's');
    end
    mkdir('forcing_term_vs_coupling_term');
    
    Ct_target   = dlmread('Ct_target.txt');
    Ct_fit      = dlmread('Ct_fit.txt');
    
    % count number of available static obstacle settings:
    i = 1;
    while (exist(num2str(i), 'dir'))
        i                   = i + 1;
    end
    num_settings            = i - 1;
    Ct_length_per_setting   = size(Ct_target, 1)/num_settings;
    
    f_baseline              = dlmread('baseline/forcing_term_trajectory.txt');
    
    for i = 1:num_settings
        start_row           = ((i-1) * Ct_length_per_setting) + 1;
        end_row             = (i     * Ct_length_per_setting);
        
        f_obs               = dlmread(strcat(num2str(i), '/forcing_term_trajectory.txt'));
        Ct_unroll_ideal     = dlmread(strcat('unroll_tests/', num2str(i),'/ideal/transform_sys_ct_acc_trajectory.txt'));
        
        figure;
        title(strcat('Demonstration Setting #',num2str(i)));
        for j = 1:3
            subplot(2,2,j);
            hold on;
            grid on;
            f_b     = plot(f_baseline(:,1),f_baseline(:,j+1), 'r');
            f_o     = plot(f_obs(:,1),f_obs(:,j+1), 'b');
            Ct_t    = plot(f_baseline(:,1),Ct_target(start_row:end_row,j), 'm');
            %err     = Ct_target(start_row:end_row,j) - (f_obs(:,j+1) - f_baseline(:,j+1));
            %mse     = err.' * err
            if (j == 1)
                title('x');
            elseif (j == 2)
                title('y');
            elseif (j == 3)
                title('z');
            end
            xlabel('time');
            ylabel('f or Ct');
            legend([f_o, f_b, Ct_t], 'f_o_b_s', 'f_b_s_l_n', 'Ct_t_a_r_g_e_t');
            hold off;
        end
        print(strcat('forcing_term_vs_coupling_term/f_obs_vs_f_baseline_setting_', num2str(i)),'-djpeg');
        
        figure;
        title(strcat('Demonstration Setting #',num2str(i)));
        for j = 1:3
            subplot(2,2,j);
            hold on;
            grid on;
            Ct_t    = plot(f_baseline(:,1),Ct_target(start_row:end_row,j), 'm');
            Ct_f    = plot(f_baseline(:,1),Ct_fit(start_row:end_row,j), 'c');
            Ct_ui   = plot(Ct_unroll_ideal(:,1),Ct_unroll_ideal(:,j+1), 'g');
            %err     = Ct_target(start_row:end_row,j) - (f_obs(:,j+1) - f_baseline(:,j+1));
            %mse     = err.' * err
            if (j == 1)
                title('x');
            elseif (j == 2)
                title('y');
            elseif (j == 3)
                title('z');
            end
            xlabel('time');
            ylabel('f or Ct');
            legend([Ct_t, Ct_f, Ct_ui], 'Ct_t_a_r_g_e_t', 'Ct_f_i_t', 'Ct_u_n_r_o_l_l_i_d_e_a_l');
            hold off;
        end
        print(strcat('forcing_term_vs_coupling_term/Ct_target_vs_Ct_fit_vs_Ct_unroll_ideal_setting_', num2str(i)),'-djpeg');
    end
end
