function [ Ct_target, Ct_fit ] = plot_learn_obs_avoid_coupling_term(  )
    % A MATLAB script to plot obstacle avoidance Ct_target and Ct_fit.
    % Author: Giovanni Sutanto
    % Date  : December 08, 2015
    close all;
    
    Ct_target   = dlmread('Ct_target.txt');
    Ct_fit      = dlmread('Ct_fit.txt');
    
    figure;
    hold on;
    plot3(Ct_target(:,1), Ct_target(:,2), Ct_target(:,3), 'r+');
    plot3(Ct_fit(:,1), Ct_fit(:,2), Ct_fit(:,3), 'co');
    title('Plot Obstacle Avoidance Ct\_target and Ct\_fit');
    xlabel('x');
    ylabel('y');
    zlabel('z');
    legend('Ct\_target','Ct\_fit');
    hold off;
end
