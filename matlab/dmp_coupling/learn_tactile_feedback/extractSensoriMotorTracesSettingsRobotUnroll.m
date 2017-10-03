function [ data_robot_unroll ] = extractSensoriMotorTracesSettingsRobotUnroll( in_data_dir_path, N_setting, N_primitive )
    data_robot_unroll.coupled   = cell(N_primitive, N_setting);
    for n_setting=1:N_setting
        setting_data_dir_path   = [in_data_dir_path, '/', num2str(n_setting,'%d'),'/'];
        if (exist(setting_data_dir_path, 'dir') == 7)
            data_robot_unroll.coupled(:,n_setting)  = extractSensoriMotorTracesSingleSetting( setting_data_dir_path );
        end
    end
end