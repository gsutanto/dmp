function [] = testCartCoordDMPFitAndUnroll(output_dir_path)
    % A MATLAB function to test 
    % Cartesian Coordinate (x-y-z) DMP 
    % with 1st order canonical system and 2nd order canonical system
    %
    % Author: Giovanni Sutanto
    % Date  : June 08, 2017
    
    close           all;
    clc;
    
    %% Data Loading
    
    % The Trajectory for "CartCoordDMP Training from Single Trajectory":
    traj_cart_coord_demo_path               = '../../../data/cart_dmp/cart_coord_dmp/single_traj_training/sample_traj_3D_1.txt';
    [ traj_cart_coord_demo, dt_0 ]          = extractSetCartCoordTrajectories( traj_cart_coord_demo_path, 2 );
    
    % The Trajectory Set #1 for "CartCoordDMP Training from Multiple Trajectories":
    set_trajs_cart_coord_demo_1_path        = '../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/baseline/endeff_trajs/';
    [ set_trajs_cart_coord_demo_1, dt_1 ]   = extractSetCartCoordTrajectories( set_trajs_cart_coord_demo_1_path, 2 );
    
    % The Trajectory Set #2 for "CartCoordDMP Training from Multiple Trajectories":
    set_trajs_cart_coord_demo_2_path        = '../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_sph_new/SubjectNo1/1/endeff_trajs/';
    [ set_trajs_cart_coord_demo_2, dt_2 ]   = extractSetCartCoordTrajectories( set_trajs_cart_coord_demo_2_path, 2 );
    
    % end of Data Loading
    
    %% DMPs Parameters Setting
    
    % params for "CartCoordDMP Training from Single Trajectory":
    n_rfs_0                                 = 50;
    
    % params for "CartCoordDMP Training from Multiple Trajectories":
    n_rfs_multi                             = 25;
    unroll_dt_multi                         = 1/420.0;
    unroll_tau_multi                        = 0.5;
    unroll_traj_length_multi                = round(unroll_tau_multi/unroll_dt_multi) + 1;
    
    ctraj_local_coordinate_frame_selection  = 1;    % gsutanto's local coordinate system
    
    % end of DMPs Parameters Setting

    %% CartCoordDMP Training from Single Trajectory
    
    [ ~, ...
      ccdmp_single_traj_training_test_0_1_unroll_global_traj ]  = learnCartPrimitiveMultiOnLocalCoord(traj_cart_coord_demo, dt_0, n_rfs_0, 0, ctraj_local_coordinate_frame_selection);
    
    [ ~, ...
      ccdmp_single_traj_training_test_0_2_unroll_global_traj ]  = learnCartPrimitiveMultiOnLocalCoord(traj_cart_coord_demo, dt_0, n_rfs_0, 1, ctraj_local_coordinate_frame_selection);
    
    % end of CartCoordDMP Training from Single Trajectory
    
    %% CartCoordDMP Training from Multiple Trajectories
    
    is_using_scaling    = [1, 0, 0];    % only DMP for local x-axis is using scaling; DMPs for local y and z-axes are NOT using scaling!
    [ ~, ...
      ccdmp_multi_trajs_training_test_0_2_t1_unroll_global_traj ]   = learnCartPrimitiveMultiOnLocalCoord(set_trajs_cart_coord_demo_1, dt_1, n_rfs_multi, 1, ctraj_local_coordinate_frame_selection, unroll_traj_length_multi, unroll_dt_multi, is_using_scaling);
    
    [ ~, ...
      ccdmp_multi_trajs_training_test_0_2_t2_unroll_global_traj ]   = learnCartPrimitiveMultiOnLocalCoord(set_trajs_cart_coord_demo_2, dt_2, n_rfs_multi, 1, ctraj_local_coordinate_frame_selection, unroll_traj_length_multi, unroll_dt_multi, is_using_scaling);
    
    % end of CartCoordDMP Training from Multiple Trajectories
    
    %% Logging Results
    
    if (exist(output_dir_path, 'dir') == 7)
        % The following applies for CartCoordDMP with 1st order canonical system:
        ccdmp_single_traj_training_test_0_1_unroll_global_position      = [ccdmp_single_traj_training_test_0_1_unroll_global_traj{4,1}-ccdmp_single_traj_training_test_0_1_unroll_global_traj{4,1}(1,1), ccdmp_single_traj_training_test_0_1_unroll_global_traj{1,1}];
        dlmwrite([output_dir_path, '/test_matlab_cart_coord_dmp_single_traj_training_test_0_1.txt'], ccdmp_single_traj_training_test_0_1_unroll_global_position, 'delimiter', ' ', 'precision', '%.5f');
        
        % The following applies for CartCoordDMP with 2nd order canonical system:
        ccdmp_single_traj_training_test_0_2_unroll_global_position      = [ccdmp_single_traj_training_test_0_2_unroll_global_traj{4,1}-ccdmp_single_traj_training_test_0_2_unroll_global_traj{4,1}(1,1), ccdmp_single_traj_training_test_0_2_unroll_global_traj{1,1}];
        dlmwrite([output_dir_path, '/test_matlab_cart_coord_dmp_single_traj_training_test_0_2.txt'], ccdmp_single_traj_training_test_0_2_unroll_global_position, 'delimiter', ' ', 'precision', '%.5f');
        
        ccdmp_multi_trajs_training_test_0_2_t1_unroll_global_position   = [ccdmp_multi_trajs_training_test_0_2_t1_unroll_global_traj{4,1}-ccdmp_multi_trajs_training_test_0_2_t1_unroll_global_traj{4,1}(1,1), ccdmp_multi_trajs_training_test_0_2_t1_unroll_global_traj{1,1}];
        ccdmp_multi_trajs_training_test_0_2_t2_unroll_global_position   = [ccdmp_multi_trajs_training_test_0_2_t2_unroll_global_traj{4,1}-ccdmp_multi_trajs_training_test_0_2_t2_unroll_global_traj{4,1}(1,1), ccdmp_multi_trajs_training_test_0_2_t2_unroll_global_traj{1,1}];
        ccdmp_multi_trajs_training_test_0_2_unroll_global_position      = [ccdmp_multi_trajs_training_test_0_2_t1_unroll_global_position; ccdmp_multi_trajs_training_test_0_2_t2_unroll_global_position];
        dlmwrite([output_dir_path, '/test_matlab_cart_coord_dmp_multi_traj_training_test_0_2.txt'], ccdmp_multi_trajs_training_test_0_2_unroll_global_position, 'delimiter', ' ', 'precision', '%.5f');
    else
        error('Output directory does NOT exist!');
    end
    
    % end of Logging Results
end