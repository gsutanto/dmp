function [] = testQuaternionDMPFitAndUnroll(output_dir_path)
    % A MATLAB function to test 
    % Quaternion DMP 
    % with 1st order canonical system and 2nd order canonical system
    %
    % Author: Giovanni Sutanto
    % Date  : Dec. 31, 2018
    
    close           all;
    clc;
    
    %% Data Loading
    
    % The Trajectory for "QuaternionDMP Training from Single Trajectory":
    traj_quat_demo_path             = '../../../data/dmp_coupling/learn_tactile_feedback/scraping_w_tool/human_baseline/prim03/07.txt';
    [ traj_quat_demo, dt_0 ]        = extractSetQuaternionTrajectories( traj_quat_demo_path, 11, 1 );
    
    % The Trajectory Set #1 for "QuaternionDMP Training from Multiple Trajectories":
    set_trajs_quat_demo_1_path      = '../../../data/dmp_coupling/learn_tactile_feedback/scraping_w_tool/human_baseline/prim03/';
    [ set_trajs_quat_demo_1, dt_1 ] = extractSetQuaternionTrajectories( set_trajs_quat_demo_1_path, 11, 1 );
    
    % The Trajectory Set #2 for "QuaternionDMP Training from Multiple Trajectories":
    set_trajs_quat_demo_2_path      = '../../../data/dmp_coupling/learn_tactile_feedback/scraping_wo_tool/human_baseline/prim02/';
    [ set_trajs_quat_demo_2, dt_2 ] = extractSetQuaternionTrajectories( set_trajs_quat_demo_2_path, 11, 1 );
    
    % end of Data Loading
    
    %% Data Smoothing
    
    percentage_padding_pose_and_joint           = 1.5;
    percentage_smoothing_points_pose_and_joint  = 3.0;
    smoothing_mode                              = 3;
    dt                                          = 1.0/300.0;
    low_pass_cutoff_freq                        = 1.5;
    
    N_traj1                                     = size(set_trajs_quat_demo_1, 2);
    set_smoothed_trajs_quat_demo_1              = cell(3, N_traj1);
    for nt1=1:N_traj1
        [ smoothed_Quat_traj1 ] = smoothStartEndQuatTrajectoryBasedOnQuaternion( set_trajs_quat_demo_1(:, nt1), ...
                                                                                 percentage_padding_pose_and_joint, ...
                                                                                 percentage_smoothing_points_pose_and_joint, ...
                                                                                 smoothing_mode, dt, ...
                                                                                 low_pass_cutoff_freq );
        set_smoothed_trajs_quat_demo_1(:,nt1)   = smoothed_Quat_traj1;
    end
    
    N_traj2                         = size(set_trajs_quat_demo_2,2);
    set_smoothed_trajs_quat_demo_2  = cell(3, N_traj2);
    for nt2=1:N_traj2
        [ smoothed_Quat_traj2 ] = smoothStartEndQuatTrajectoryBasedOnQuaternion( set_trajs_quat_demo_2(:, nt2), ...
                                                                                 percentage_padding_pose_and_joint, ...
                                                                                 percentage_smoothing_points_pose_and_joint, ...
                                                                                 smoothing_mode, dt, ...
                                                                                 low_pass_cutoff_freq );
        set_smoothed_trajs_quat_demo_2(:,nt2)   = smoothed_Quat_traj2;
    end
    
    % end of Data Smoothing
    
    %% DMPs Parameters Setting
    
    unroll_tau                              = 1.9976;
    % params for "QuaternionDMP Training from Single Trajectory":
    n_rfs_0                                 = 50;
    
    % params for "QuaternionDMP Training from Multiple Trajectories":
    n_rfs_multi                             = 25;
    
    % end of DMPs Parameters Setting

    %% QuaternionDMP Training from Single Trajectory
    
    [ ~, ...
      quatdmp_single_traj_training_test_0_1_unroll_traj ]    = learnQuatPrimitiveMulti(traj_quat_demo, dt_0, n_rfs_0, 0, unroll_tau);
    
    [ ~, ...
      quatdmp_single_traj_training_test_0_2_unroll_traj ]    = learnQuatPrimitiveMulti(traj_quat_demo, dt_0, n_rfs_0, 1, unroll_tau);
    
    % end of QuaternionDMP Training from Single Trajectory
    
    %% QuaternionDMP Training from Multiple Trajectories
    
    [ Quat_dmp_params_1, ...
      quatdmp_multi_trajs_training_test_0_2_t1_unroll_traj ] = learnQuatPrimitiveMulti(set_trajs_quat_demo_1, dt_1, n_rfs_multi, 1, unroll_tau);
    
    [ Quat_dmp_params_2, ...
      quatdmp_multi_trajs_training_test_0_2_t2_unroll_traj ] = learnQuatPrimitiveMulti(set_trajs_quat_demo_2, dt_2, n_rfs_multi, 1, unroll_tau);
    
    % end of QuaternionDMP Training from Multiple Trajectories
    
    %% QuaternionDMP Training from Multiple Smoothed Trajectories
    
    [ Quat_dmp_smoothed_params_1, ...
      quatdmp_multi_smoothed_trajs_training_test_0_2_t1_unroll_traj ] = learnQuatPrimitiveMulti(set_smoothed_trajs_quat_demo_1, dt_1, n_rfs_multi, 1, unroll_tau);
    
    [ Quat_dmp_smoothed_params_2, ...
      quatdmp_multi_smoothed_trajs_training_test_0_2_t2_unroll_traj ] = learnQuatPrimitiveMulti(set_smoothed_trajs_quat_demo_2, dt_2, n_rfs_multi, 1, unroll_tau);
    
    % end of QuaternionDMP Training from Multiple Smoothed Trajectories
    
    %% Logging Results
    
    if (exist(output_dir_path, 'dir') == 7)
        % The following applies for QuaternionDMP with 1st order canonical system:
        quatdmp_single_traj_training_test_0_1_unroll_Q      = [quatdmp_single_traj_training_test_0_1_unroll_traj{6,1}-quatdmp_single_traj_training_test_0_1_unroll_traj{6,1}(1,1), quatdmp_single_traj_training_test_0_1_unroll_traj{1,1}];
        dlmwrite([output_dir_path, '/test_matlab_quat_dmp_single_traj_training_test_0_1.txt'], quatdmp_single_traj_training_test_0_1_unroll_Q, 'delimiter', ' ', 'precision', '%.10f');
        
        % The following applies for QuaternionDMP with 2nd order canonical system:
        quatdmp_single_traj_training_test_0_2_unroll_Q      = [quatdmp_single_traj_training_test_0_2_unroll_traj{6,1}-quatdmp_single_traj_training_test_0_2_unroll_traj{6,1}(1,1), quatdmp_single_traj_training_test_0_2_unroll_traj{1,1}];
        dlmwrite([output_dir_path, '/test_matlab_quat_dmp_single_traj_training_test_0_2.txt'], quatdmp_single_traj_training_test_0_2_unroll_Q, 'delimiter', ' ', 'precision', '%.10f');
        
        quatdmp_multi_trajs_training_test_0_2_t1_unroll_Q   = [quatdmp_multi_trajs_training_test_0_2_t1_unroll_traj{6,1}-quatdmp_multi_trajs_training_test_0_2_t1_unroll_traj{6,1}(1,1), quatdmp_multi_trajs_training_test_0_2_t1_unroll_traj{1,1}];
        quatdmp_multi_trajs_training_test_0_2_t2_unroll_Q   = [quatdmp_multi_trajs_training_test_0_2_t2_unroll_traj{6,1}-quatdmp_multi_trajs_training_test_0_2_t2_unroll_traj{6,1}(1,1), quatdmp_multi_trajs_training_test_0_2_t2_unroll_traj{1,1}];
        quatdmp_multi_trajs_training_test_0_2_unroll_Q      = [quatdmp_multi_trajs_training_test_0_2_t1_unroll_Q; 
                                                               quatdmp_multi_trajs_training_test_0_2_t2_unroll_Q];
        dlmwrite([output_dir_path, '/test_matlab_quat_dmp_multi_traj_training_test_0_2.txt'], quatdmp_multi_trajs_training_test_0_2_unroll_Q, 'delimiter', ' ', 'precision', '%.10f');
        
        quatdmp_multi_trajs_training_test_0_2_unroll_learned_params = [Quat_dmp_params_1.w, zeros(n_rfs_multi, 2); 
                                                                       Quat_dmp_params_1.dG.', Quat_dmp_params_1.fit_mean_tau, 0.0; 
                                                                       Quat_dmp_params_1.fit_mean_Q0.', 0.0; 
                                                                       Quat_dmp_params_1.fit_mean_QG.', 0.0; 
                                                                       Quat_dmp_params_2.w, zeros(n_rfs_multi, 2); 
                                                                       Quat_dmp_params_2.dG.', Quat_dmp_params_2.fit_mean_tau, 0.0; 
                                                                       Quat_dmp_params_2.fit_mean_Q0.', 0.0; 
                                                                       Quat_dmp_params_2.fit_mean_QG.', 0.0];
        dlmwrite([output_dir_path, '/test_matlab_quat_dmp_multi_traj_training_test_0_2_learned_params.txt'], quatdmp_multi_trajs_training_test_0_2_unroll_learned_params, 'delimiter', ' ', 'precision', '%.10f');
        
        quatdmp_multi_smoothed_trajs_training_test_0_2_t1_unroll_Q   = [quatdmp_multi_smoothed_trajs_training_test_0_2_t1_unroll_traj{6,1}-quatdmp_multi_smoothed_trajs_training_test_0_2_t1_unroll_traj{6,1}(1,1), quatdmp_multi_smoothed_trajs_training_test_0_2_t1_unroll_traj{1,1}];
        quatdmp_multi_smoothed_trajs_training_test_0_2_t2_unroll_Q   = [quatdmp_multi_smoothed_trajs_training_test_0_2_t2_unroll_traj{6,1}-quatdmp_multi_smoothed_trajs_training_test_0_2_t2_unroll_traj{6,1}(1,1), quatdmp_multi_smoothed_trajs_training_test_0_2_t2_unroll_traj{1,1}];
        quatdmp_multi_smoothed_trajs_training_test_0_2_unroll_Q      = [quatdmp_multi_smoothed_trajs_training_test_0_2_t1_unroll_Q; 
                                                                        quatdmp_multi_smoothed_trajs_training_test_0_2_t2_unroll_Q];
        dlmwrite([output_dir_path, '/test_matlab_quat_dmp_multi_smoothed_traj_training_test_0_2.txt'], quatdmp_multi_smoothed_trajs_training_test_0_2_unroll_Q, 'delimiter', ' ', 'precision', '%.10f');
        
        qdmp_multi_smoothed_trajs_training_test_0_2_unroll_lparams = [Quat_dmp_smoothed_params_1.w, zeros(n_rfs_multi, 2); 
                                                                      Quat_dmp_smoothed_params_1.dG.', Quat_dmp_smoothed_params_1.fit_mean_tau, 0.0; 
                                                                      Quat_dmp_smoothed_params_1.fit_mean_Q0.', 0.0; 
                                                                      Quat_dmp_smoothed_params_1.fit_mean_QG.', 0.0; 
                                                                      Quat_dmp_smoothed_params_2.w, zeros(n_rfs_multi, 2); 
                                                                      Quat_dmp_smoothed_params_2.dG.', Quat_dmp_smoothed_params_2.fit_mean_tau, 0.0; 
                                                                      Quat_dmp_smoothed_params_2.fit_mean_Q0.', 0.0; 
                                                                      Quat_dmp_smoothed_params_2.fit_mean_QG.', 0.0];
        dlmwrite([output_dir_path, '/test_matlab_quat_dmp_multi_smoothed_traj_training_test_0_2_learned_params.txt'], qdmp_multi_smoothed_trajs_training_test_0_2_unroll_lparams, 'delimiter', ' ', 'precision', '%.10f');
    else
        error('Output directory does NOT exist!');
    end
    
    % end of Logging Results
end