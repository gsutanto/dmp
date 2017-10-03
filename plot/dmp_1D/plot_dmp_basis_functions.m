function [  ] = plot_dmp_basis_functions(  )
    % a simple MATLAB script to plot recorded DMP basis functions
    % Author: Giovanni Sutanto
    % Date  : June 19, 2015
    close all;
    
%     Previous Implementation:
%     sample_dmp_basis_functions = csvread('sample_dmp_basis_functions.txt');
%     plot(sample_dmp_basis_functions(:,1),sample_dmp_basis_functions(:,[1:end]))
%     title('Plot of DMP Basis Functions');
%     xlabel('canonical state position');
%     ylabel('basis function magnitude');

    basis_functions_trajectory = dlmread('basis_functions_trajectory.txt');
    
    plot(basis_functions_trajectory(:,1), basis_functions_trajectory(:,[2:end]))
    title('Plot of DMP Basis Functions');
    xlabel('time');
    ylabel('basis function magnitude');
end