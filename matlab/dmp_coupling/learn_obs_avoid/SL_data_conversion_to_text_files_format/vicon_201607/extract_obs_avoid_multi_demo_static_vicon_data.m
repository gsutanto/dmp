% Author: Giovanni Sutanto
% Date  : July 17, 2016
close all;
clc;

addpath([pwd,'/../../../../utilities/']);
addpath([pwd,'/../../../../utilities/quaternion/']);
addpath([pwd,'/../../../../utilities/clmcplot/']);
addpath([pwd,'/../../utilities/']);
addpath([pwd,'/../../vicon/vicon_objects/']);
addpath([pwd,'/../utilities/']);
addpath([pwd,'/../utilities/']);

% Change in_data_dir_path to the path of the input data directory, as
% necessary:
% in_data_dir_path    = '/home/gsutanto/Desktop/CLMC/Data/DMP_LOA_Ct_Vicon_Data_Collection_201607/learn_obs_avoid_gsutanto_vicon_data/';
in_data_dir_path    = '/media/GSUTANTO/My Documents/USC/Research/CLMC/Data/DMP_LOA_Ct_Vicon_Data_Collection_201607/learn_obs_avoid_gsutanto_vicon_data/';
out_data_dir_path   = '../../../../../data/dmp_coupling/learn_obs_avoid/static_obs/';

origin_folder       = pwd;
addpath(origin_folder);
cd(in_data_dir_path);

obs_geometry        = {'sphere', 'cube', 'cyl'};

out_data_dir_name   = 'data_multi_demo_vicon_static';

% If output directory is already exist, then remove it and re-create one:
if (exist(out_data_dir_name, 'dir'))
    rmdir(out_data_dir_name, 's');
end
mkdir(out_data_dir_name);

extract_obs_avoid_single_demo_static_vicon_data('baseline', out_data_dir_name, 0);

overall_demo_setting_count              = 1;

for i=1:size(obs_geometry, 2)
    per_object_demo_setting_count       = 1;
    in_data_subdir_name     = [obs_geometry{1,i},'_static_obs/',num2str(per_object_demo_setting_count),'/'];
    while (exist(in_data_subdir_name, 'dir'))
        extract_obs_avoid_single_demo_static_vicon_data(in_data_subdir_name, out_data_dir_name, overall_demo_setting_count);
        per_object_demo_setting_count   = per_object_demo_setting_count + 1;
        overall_demo_setting_count      = overall_demo_setting_count + 1;
        in_data_subdir_name = [obs_geometry{1,i},'_static_obs/',num2str(per_object_demo_setting_count),'/'];
    end
end

out_data_dir_fullpath       = [origin_folder, '/', out_data_dir_path, '/', out_data_dir_name];
% If output directory is already exist, then remove it and re-create one:
if (exist(out_data_dir_fullpath, 'dir'))
    rmdir(out_data_dir_fullpath, 's');
end
movefile(out_data_dir_name, [origin_folder, '/', out_data_dir_path]);
cd(origin_folder);
rmpath(origin_folder);