function [] = testQuaternionDMPUnroll(output_dir_path)
    % A MATLAB function to test unrolling a Quaternion DMP 
    % with loaded parameters.
    %
    % Author: Giovanni Sutanto
    % Date  : June 23, 2017
    
    close           all;
    clc;
    
    %% Quaternion DMP Parameters Setting and Loading
    
    Quat_dmp_params.dt  	= 1/300.0;
    Quat_dmp_params.n_rfs 	= 25;
    Quat_dmp_params.c_order = 1;
    
    % Directory path containing a pre-trained Quaternion DMP and additional parameters:
    quat_dmp_data_path              = '../../../data/cart_dmp/quat_dmp/';
    
    Quat_dmp_params.w               = dlmread([quat_dmp_data_path, 'w']).';
    Quat_dmp_params.dG              = dlmread([quat_dmp_data_path, 'A_learn']);
    
    unroll_Quat_params.tau          = dlmread([quat_dmp_data_path, 'tau']);
    unroll_Quat_params.traj_length  = -1;
    unroll_Quat_params.start        = dlmread([quat_dmp_data_path, 'Q0']);
    unroll_Quat_params.goal         = dlmread([quat_dmp_data_path, 'QG']);
    unroll_Quat_params.omega0       = zeros(3, 1);
    unroll_Quat_params.omegad0      = zeros(3, 1);
    
    % end of Quaternion DMP Parameters Setting and Loading

    %% Quaternion DMP Unrolling
    
    % with zero initial angular velocity and zero initial angular
    % acceleration:
    [Quat_dmp_unroll_traj_0]        = unrollQuatPrimitive( Quat_dmp_params, ...
                                                           unroll_Quat_params );
    
    % with NON-zero initial angular velocity and zero initial angular
    % acceleration, and duration (tau) 1.5 times longer:
    unroll_Quat_params.omega0       = dlmread([quat_dmp_data_path, 'nonzero_omega']);
    unroll_Quat_params.omegad0      = dlmread([quat_dmp_data_path, 'nonzero_omegad']);
    unroll_Quat_params.tau          = 1.5 * unroll_Quat_params.tau;
    [Quat_dmp_unroll_traj_1]        = unrollQuatPrimitive( Quat_dmp_params, ...
                                                           unroll_Quat_params );
    
    % end of Quaternion DMP Unrolling
    
    %% Logging Results
    
    if (exist(output_dir_path, 'dir') == 7)
        % The following applies for Quaternion DMP 
        % with zero initial angular velocity and zero initial angular
        % acceleration:
        qdmp_unroll_result_combined_0   = [Quat_dmp_unroll_traj_0{6,1}-Quat_dmp_unroll_traj_0{6,1}(1,1), ...
                                           Quat_dmp_unroll_traj_0{1,1}, ...
                                           Quat_dmp_unroll_traj_0{2,1}, ...
                                           Quat_dmp_unroll_traj_0{3,1}, ...
                                           Quat_dmp_unroll_traj_0{4,1}, ...
                                           Quat_dmp_unroll_traj_0{5,1}];
        
        % The following applies for Quaternion DMP 
        % with NON-zero initial angular velocity and zero initial angular
        % acceleration, and duration (tau) 1.5 times longer:
        qdmp_unroll_result_combined_1   = [Quat_dmp_unroll_traj_1{6,1}-Quat_dmp_unroll_traj_1{6,1}(1,1), ...
                                           Quat_dmp_unroll_traj_1{1,1}, ...
                                           Quat_dmp_unroll_traj_1{2,1}, ...
                                           Quat_dmp_unroll_traj_1{3,1}, ...
                                           Quat_dmp_unroll_traj_1{4,1}, ...
                                           Quat_dmp_unroll_traj_1{5,1}];
        
        qdmp_unroll_result_combined_all = [qdmp_unroll_result_combined_0; ...
                                           qdmp_unroll_result_combined_1];
        dlmwrite([output_dir_path, '/test_matlab_quat_dmp_unroll_test.txt'], qdmp_unroll_result_combined_all(:, 1:5), 'delimiter', ' ', 'precision', '%.5f');
    else
        error('Output directory does NOT exist!');
    end
    
    % end of Logging Results
end