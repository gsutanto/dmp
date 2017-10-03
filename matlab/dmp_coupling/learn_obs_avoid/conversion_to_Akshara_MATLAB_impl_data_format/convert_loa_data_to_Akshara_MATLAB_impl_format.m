function [ data ] = convert_loa_data_to_Akshara_MATLAB_impl_format( dataset_name, save_name, version )
    % Author: Giovanni Sutanto
    % Date  : January 27, 2016
    % 
    % Example Usage: 
    % [ data ] = convert_loa_data_to_Akshara_MATLAB_impl_format( 'data_multi_demo_static','data_multi_demo_static', 0 );
    
    close   all;
    clc;

    addpath('../utilities/');
    
    dir_path        = ['../../../../data/dmp_coupling/learn_obs_avoid/static_obs/', dataset_name, '/'];
    baseline        = extractSetCartCoordTrajectories([dir_path, 'baseline/endeff_trajs/'], version);
    i               = 1;
    subdir_path     = [dir_path, num2str(i), '/'];
    while (exist(subdir_path, 'dir'))
        data{i,1}   = baseline;
        data{i,2}   = dlmread([subdir_path, 'obs_sph_center_coord.txt'])';
        w_obs       = extractSetCartCoordTrajectories([subdir_path, 'endeff_trajs/'], version);
        data{i,3}   = w_obs;
        if (version == 2)
            data{i,4}   = dlmread([subdir_path, 'obs_sph_radius.txt'])';
        end
        i           = i+1;
        subdir_path = [dir_path, num2str(i), '/'];
    end

    save([save_name, '.mat'], 'data');
end