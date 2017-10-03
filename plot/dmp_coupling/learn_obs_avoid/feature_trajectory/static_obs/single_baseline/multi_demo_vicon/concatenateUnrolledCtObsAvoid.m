function [ Ct_unroll ] = concatenateUnrolledCtObsAvoid( data_dir_path )
    close all;
    clc;

    in_dir_path                     = [data_dir_path, 'unroll_tests/'];

    Ct_unroll                       = [];

    setting_count                   = 1;
    in_subdir_path                  = [in_dir_path,num2str(setting_count),'/'];
    while (exist(in_subdir_path, 'dir'))
        disp(['Processing Setting #',num2str(setting_count)]);
        demo_per_setting_count      = 1;
        in_subsubdir_path           = [in_subdir_path,num2str(demo_per_setting_count),'/'];
        while (exist(in_subsubdir_path, 'dir'))
            disp(['   Processing Demo #',num2str(demo_per_setting_count)]);
            ct_oa_file_path         = [in_subsubdir_path,'transform_sys_ct_acc_trajectory.txt'];
            ct_oa_traj              = dlmread(ct_oa_file_path);
            ct_oa                   = ct_oa_traj(:,2:4);
            Ct_unroll               = [Ct_unroll; ct_oa];
            demo_per_setting_count  = demo_per_setting_count + 1;
            in_subsubdir_path       = [in_subdir_path,num2str(demo_per_setting_count),'/'];
        end
        setting_count               = setting_count + 1;
        in_subdir_path              = [in_dir_path,num2str(setting_count),'/'];
    end

    dlmwrite([data_dir_path, 'Ct_unroll.txt'], Ct_unroll, 'delimiter', ' ');
end