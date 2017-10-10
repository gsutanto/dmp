function [ data_global_coord ] = prepareDemoDatasetLOAVicon(  )
    %% Author: Giovanni Sutanto
    
    trajs_extraction_version        = 2;
    
    dir_path                        = ['../../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_vicon_static/'];
    data_global_coord.baseline      = extractSetCartCoordTrajectories([dir_path, 'baseline/endeff_trajs/'], trajs_extraction_version);
    obs_avoid_dominant_axis_annotation      = dlmread([dir_path, 'data_annotation_obs_avoid_dominant_axis.txt']);
    obs_avoid_demo_consistency_annotation   = dlmread([dir_path, 'data_annotation_obs_avoid_consistency.txt']);

    freq                            = 300.0;
    data_global_coord.dt            = 1/freq;
    data_global_coord.obs_avoid_var_descriptor{1,1} = ['obs_markers_global_coord'];
    data_global_coord.obs_avoid_var_descriptor{1,2} = ['endeff_trajs'];
    data_global_coord.obs_avoid_var_descriptor{1,3} = ['obs_avoid_dominant_axis'];
    data_global_coord.obs_avoid_var_descriptor{1,4} = ['obs_avoid_demo_consistency'];
    i                               = 1;
    subdir_path                     = [dir_path, num2str(i), '/'];
    while (exist(subdir_path, 'dir'))
        data_global_coord.obs_avoid{i,1}    = dlmread([subdir_path, data_global_coord.obs_avoid_var_descriptor{1,1}, '.txt']);
        data_global_coord.obs_avoid{i,2}    = extractSetCartCoordTrajectories([subdir_path, data_global_coord.obs_avoid_var_descriptor{1,2}, '/'], trajs_extraction_version);
        data_global_coord.obs_avoid{i,3}    = obs_avoid_dominant_axis_annotation(i,1);
        data_global_coord.obs_avoid{i,4}    = any(i==obs_avoid_demo_consistency_annotation);
        i                                   = i+1;
        subdir_path                         = [dir_path, num2str(i), '/'];
    end
end