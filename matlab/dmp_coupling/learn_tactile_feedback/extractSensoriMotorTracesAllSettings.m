function [ data_demo ] = extractSensoriMotorTracesAllSettings( in_data_dir_path )
    baseline_data_dir_path  = [in_data_dir_path, '/baseline/'];
    data_demo.baseline      = extractSensoriMotorTracesSingleSetting( baseline_data_dir_path );
    N_primitive             = size(data_demo.baseline, 1);
    
    % Count how many settings there are:
    N_setting                   = 1;    % # of settings
    setting_data_dir_path       = [in_data_dir_path, '/', num2str(N_setting,'%d'),'/'];
    while (exist(setting_data_dir_path, 'dir') == 7)
        data_files              = dir(strcat(setting_data_dir_path,'/','j*'));
        if (size(data_files, 1) == 1)
            N_setting               = N_setting + 1;
            setting_data_dir_path   = [in_data_dir_path, '/', num2str(N_setting,'%d'),'/'];
        else
            break;
        end
    end
    N_setting                   = N_setting - 1;
    
    data_demo.coupled           = cell(N_primitive, N_setting);
    for n_setting=1:N_setting
        setting_data_dir_path   = [in_data_dir_path, '/', num2str(n_setting,'%d'),'/'];
        data_demo.coupled(:,n_setting)  = extractSensoriMotorTracesSingleSetting( setting_data_dir_path );
    end
end