function plot_loa_Ct_target_vs_Ct_fit_per_setting(  )
    % A MATLAB script to plot obstacle avoidance Ct_target versus Ct_fit 
    % on different static obstacle settings.
    % Author: Giovanni Sutanto
    % Date  : January 05, 2016
    close all;
    clc;
    
    Ct_target   = dlmread('Ct_target.txt');
    Ct_fit      = dlmread('Ct_fit.txt');
    
    % count number of available static obstacle settings:
    i = 1;
    while (exist(num2str(i), 'dir'))
        i                   = i + 1;
    end
    num_settings            = i - 1;
    Ct_length_per_setting   = size(Ct_target, 1)/num_settings;
    
    for i = 1:num_settings
        start_row           = ((i-1) * Ct_length_per_setting) + 1;
        end_row             = (i     * Ct_length_per_setting);
        figure;
        hold on;
        plot3(Ct_target(start_row:end_row,1), Ct_target(start_row:end_row,2), Ct_target(start_row:end_row,3), 'r+');
        plot3(Ct_fit(start_row:end_row,1), Ct_fit(start_row:end_row,2), Ct_fit(start_row:end_row,3), 'co');
        title(strcat('Plot Obstacle Avoidance Ct\_target vs Ct\_fit on Setting #', num2str(i)));
        xlabel('x');
        ylabel('y');
        zlabel('z');
        legend('Ct\_target','Ct\_fit');
        hold off;
    end
end
