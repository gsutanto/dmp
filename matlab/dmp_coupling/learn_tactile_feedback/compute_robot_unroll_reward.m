% Author: Giovanni Sutanto
% Date  : May 8, 2019

close all;
clear all;
% clc;

addpath('../../../matlab/utilities/');
addpath('../../../matlab/utilities/clmcplot/');

generic_task_type       = 'scraping';
specific_task_type      = 'scraping_w_tool';
date                    = '20190508';
additional_description  = '';
experiment_name         = [date,'_',specific_task_type,'_correctable',additional_description];

homepath                = getenv('HOME');
N_total_sense_dimensionality    = 45;

env_setting_name        = 'n5';
unroll_types            = {'b', 'c'};   % 'b' = baseline; 'c' = coupled
prim2_Rewards           = cell(size(unroll_types));
prim3_Rewards           = cell(size(unroll_types));
prim2_aveRewards        = cell(size(unroll_types));
prim3_aveRewards        = cell(size(unroll_types));
disp(['Environment Setting ', env_setting_name]);
for uti = 1:length(unroll_types)
    data_dir_path       = [homepath,'/Desktop/dmp_robot_unroll_results/scraping/',experiment_name,'/robot/',env_setting_name,'/',unroll_types{1,uti}];
    data_files          = dir([data_dir_path,'/d*']);
    N_files             = length(data_files);
    prim2_Rewards{1,uti}= zeros(N_files, 1);
    prim3_Rewards{1,uti}= zeros(N_files, 1);
    n_file              = 0;
    for data_file = data_files'
        n_file          = n_file + 1;
        data_file_path  = [data_dir_path,'/',data_file.name];
        [D,vars,freq]   = clmcplot_convert(data_file_path);
        vector          = cell(1,N_total_sense_dimensionality);
        for i = 1:N_total_sense_dimensionality
            vector{1,i} = clmcplot_getvariables(D, vars, {strcat('X_vector_',num2str(i-1,'%02d'))});
        end
        X_vector        = horzcat(vector{:});

        np              = clmcplot_getvariables(D, vars, {'ul_curr_prim_no'});
        prim2_indices   = find(np == 1);
        prim3_indices   = find(np == 2);

        prim2_X_vector  = X_vector(prim2_indices,:);
        prim3_X_vector  = X_vector(prim3_indices,:);

        prim2_Rewards{1,uti}(n_file,1)  = -norm(prim2_X_vector);
        prim3_Rewards{1,uti}(n_file,1)  = -norm(prim3_X_vector);
    end
    prim2_aveRewards{1,uti}  = mean(prim2_Rewards{1,uti});
    prim3_aveRewards{1,uti}  = mean(prim3_Rewards{1,uti});
    disp(['   Unroll Type ', unroll_types{1,uti}]);
    disp(['      Prim. 2 Average Reward = ', num2str(prim2_aveRewards{1,uti})]);
    disp(['      Prim. 3 Average Reward = ', num2str(prim3_aveRewards{1,uti})]);
end