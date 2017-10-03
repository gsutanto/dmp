function extract_obs_avoid_multi_demo_static_data()
    % Author: Giovanni Sutanto
    % Date  : December 26, 2015
    close all;
    clc;
    
    addpath([pwd,'/../../../../utilities/clmcplot/']);
    addpath([pwd,'/../utilities/']);
    
    % Change in_data_dir_path to the path of the input data directory, as
    % necessary:
    in_data_dir_path    = '/home/gsutanto/prog/masterUser/dmp_related/dmp_coupling/learn_obs_avoid/20151226_demo_obs_avoid_gsutanto';
    out_data_dir_path   = '../../../../../data/dmp_coupling/learn_obs_avoid/static_obs/';
    
    origin_folder       = pwd;
    addpath(origin_folder);
    cd(in_data_dir_path);

    dir_names           = {'baseline', '1', '2', '3', '4', '5', '6', '7'};
    out_data_dir_name   = 'data_multi_demo_static';
    
    if (exist(out_data_dir_name, 'dir'))
        rmdir(out_data_dir_name, 's');
    end
    mkdir(out_data_dir_name);
    
    extract_obs_avoid_single_demo_static_data('baseline', out_data_dir_name);
    
    demo_setting_count      = 1;
    while (exist(num2str(demo_setting_count), 'dir'))
        extract_obs_avoid_single_demo_static_data(num2str(demo_setting_count), out_data_dir_name);
        demo_setting_count  = demo_setting_count + 1;
    end
    
    movefile(out_data_dir_name, [origin_folder, '/', out_data_dir_path]);
    cd(origin_folder);
    rmpath(origin_folder);
end

