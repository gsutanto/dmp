function [ data_demo_baseline ] = extractNominalSensoriMotorTraces( in_data_dir_path )
    baseline_data_dir_path  = [in_data_dir_path, '/human_baseline/'];
    data_demo_baseline      = extractSensoriMotorTracesSingleSetting( baseline_data_dir_path );
end