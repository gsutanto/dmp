% Author: Giovanni Sutanto
% Date  : Jan 30, 2017

close all;
clear all;
clc;

addpath('../../../matlab/utilities/');
addpath('../../../matlab/utilities/clmcplot/');
addpath('../../../matlab/utilities/quaternion/');

task_type           = 'peg_in_hole_big_cone';

% Change in_data_dir_path to the path of the input data directory, as
% necessary:
in_data_dir_path    = ['~/Desktop/dmp_demos/peg_in_hole/',task_type,'/'];
out_data_dir_path   = [pwd, '/',task_type,'/'];

is_performing_transformation_to_object_peg_board_coord_sys  = 1;

% peg_tool is the tip of the transformation chain       (is_peg_tool_tip_of_transform_chain == 1)
% end-effector is the tip of the transformation chain   (is_peg_tool_tip_of_transform_chain == 0)
is_peg_tool_tip_of_transform_chain                          = 0;

is_plotting         = 0;

contents    = dir(in_data_dir_path);
for iter_obj = contents'
    in_iter_obj_path    = [in_data_dir_path, iter_obj.name];
    if ((strcmp(iter_obj.name, '.') == 0) && (strcmp(iter_obj.name, '..') == 0)...
        && (isdir(in_iter_obj_path)))
        
        out_iter_obj_path   = [out_data_dir_path, iter_obj.name];
        recreateDir(out_iter_obj_path);
        data_files      = dir(strcat(in_iter_obj_path,'/','d*'));
        data_file_count = 1;
        for data_file = data_files'
            in_data_file_path  = [in_iter_obj_path,'/',data_file.name];
            fprintf('Processing %s...\n', in_data_file_path);
            [D,vars,freq]   = clmcplot_convert(in_data_file_path);

            time    = clmcplot_getvariables(D, vars, {'time'}); % [0.0:(1/420.0):4.0-(1/420.0)];
            [traj_x, traj_y, traj_z]        = clmcplot_getvariables(D, vars, {'R_HAND_x','R_HAND_y','R_HAND_z'});
            [traj_xd, traj_yd, traj_zd]     = clmcplot_getvariables(D, vars, {'R_HAND_xd','R_HAND_yd','R_HAND_zd'});
            [traj_xdd, traj_ydd, traj_zdd]  = clmcplot_getvariables(D, vars, {'R_HAND_xdd','R_HAND_ydd','R_HAND_zdd'});
            [traj_q0, traj_q1, traj_q2, traj_q3]        = clmcplot_getvariables(D, vars, {'R_HAND_q0','R_HAND_q1','R_HAND_q2','R_HAND_q3'});
%             [traj_q0d, traj_q1d, traj_q2d, traj_q3d]    = clmcplot_getvariables(D, vars, {'R_HAND_q0d','R_HAND_q1d','R_HAND_q2d','R_HAND_q3d'});
%             [traj_q0dd, traj_q1dd, traj_q2dd, traj_q3dd]= clmcplot_getvariables(D, vars, {'R_HAND_q0dd','R_HAND_q1dd','R_HAND_q2dd','R_HAND_q3dd'});
            [traj_ad,  traj_bd,  traj_gd]   = clmcplot_getvariables(D, vars, {'R_HAND_ad','R_HAND_bd','R_HAND_gd'});
            [traj_add, traj_bdd, traj_gdd]  = clmcplot_getvariables(D, vars, {'R_HAND_add','R_HAND_bdd','R_HAND_gdd'});
%             cart_coord_traj         = [time, traj_x, traj_y, traj_z, traj_xd, traj_yd, traj_zd, traj_xdd, traj_ydd, traj_zdd];
%             cart_quat_traj          = [time, traj_q0, traj_q1, traj_q2, traj_q3, traj_q0d, traj_q1d, traj_q2d, traj_q3d, traj_q0dd, traj_q1dd, traj_q2dd, traj_q3dd];
%             cart_quat_ABGomega_traj = [time, traj_q0, traj_q1, traj_q2, traj_q3, traj_ad,  traj_bd,  traj_gd, traj_add, traj_bdd, traj_gdd];
            [traj_R_Fx, traj_R_Fy, traj_R_Fz] = clmcplot_getvariables(D, vars, {'R_HAND_d_locFiltFX','R_HAND_d_locFiltFY','R_HAND_d_locFiltFZ'});
            [traj_R_Tx, traj_R_Ty, traj_R_Tz] = clmcplot_getvariables(D, vars, {'R_HAND_d_locFiltTX','R_HAND_d_locFiltTY','R_HAND_d_locFiltTZ'});
            traj_R_LF_electrodes    = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_LF_','E',[1:19]));
            traj_R_LF_TDC   = clmcplot_getvariables(D, vars, {'R_LF_TDC'});
            traj_R_LF_TAC   = clmcplot_getvariables(D, vars, {'R_LF_TAC'});
            traj_R_LF_PDC   = clmcplot_getvariables(D, vars, {'R_LF_PDC'});
            traj_R_LF_PACs  = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_LF_','PAC_',[1:22]));
            [traj_R_LF_FX, traj_R_LF_FY, traj_R_LF_FZ]  = clmcplot_getvariables(D, vars, {'R_LF_FX', 'R_LF_FY', 'R_LF_FZ'});
            [traj_R_LF_MX, traj_R_LF_MY, traj_R_LF_MZ]  = clmcplot_getvariables(D, vars, {'R_LF_MX', 'R_LF_MY', 'R_LF_MZ'});
            [traj_R_LF_POC_X, traj_R_LF_POC_Y, traj_R_LF_POC_Z]  = clmcplot_getvariables(D, vars, {'R_LF_POC_X', 'R_LF_POC_Y', 'R_LF_POC_Z'});
            [traj_object_peg_board_xyz]        = clmcplot_getvariables(D, vars, {'object_peg_board_x','object_peg_board_y','object_peg_board_z'});
            [traj_object_peg_board_qwxyz]      = clmcplot_getvariables(D, vars, {'object_peg_board_qw','object_peg_board_qx','object_peg_board_qy','object_peg_board_qz'});
            [traj_peg_tool_xyz]     = clmcplot_getvariables(D, vars, {'peg_tool_x','peg_tool_y','peg_tool_z'});
            [traj_peg_tool_qwxyz]   = clmcplot_getvariables(D, vars, {'peg_tool_qw','peg_tool_qx','peg_tool_qy','peg_tool_qz'});
            traj_R_RF_electrodes    = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_RF_','E',[1:19]));
            traj_R_RF_TDC   = clmcplot_getvariables(D, vars, {'R_RF_TDC'});
            traj_R_RF_TAC   = clmcplot_getvariables(D, vars, {'R_RF_TAC'});
            traj_R_RF_PDC   = clmcplot_getvariables(D, vars, {'R_RF_PDC'});
            traj_R_RF_PACs  = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_RF_','PAC_',[1:22]));
            [traj_R_RF_FX, traj_R_RF_FY, traj_R_RF_FZ]  = clmcplot_getvariables(D, vars, {'R_RF_FX', 'R_RF_FY', 'R_RF_FZ'});
            [traj_R_RF_MX, traj_R_RF_MY, traj_R_RF_MZ]  = clmcplot_getvariables(D, vars, {'R_RF_MX', 'R_RF_MY', 'R_RF_MZ'});
            [traj_R_RF_POC_X, traj_R_RF_POC_Y, traj_R_RF_POC_Z]  = clmcplot_getvariables(D, vars, {'R_RF_POC_X', 'R_RF_POC_Y', 'R_RF_POC_Z'});
            
            traj_endeff_velocity    = [traj_xd, traj_yd, traj_zd];
                   
            [ retain_idx_cell ] = getDataClippingRetainIndex( ...
                                        traj_endeff_velocity, is_plotting, ...
                                        0.12, 0.035, 3);
            
            if (data_file_count == 1)
                traj_joints     = clmcplot_getvariables(D, vars, ...
                                    getDataNamesFromCombinations('R_', ...
                                        {'SFE_', 'SAA_', 'HR_', 'EB_', 'WR_', 'WFE_', 'WAA_', 'FR_', 'RF_', 'MF_', 'LF_'}, ...
                                        {'th', 'thd', 'thdd'}));
                joint_start_states  = [time(1,1), traj_joints(1,:)];
                out_iter_obj_joint_start_states_file_path   = [out_iter_obj_path,'/joint_start_states.txt'];
                dlmwrite(out_iter_obj_joint_start_states_file_path, joint_start_states, ...
                         'delimiter', ' ');
            end
            
            sm_traj = [time  ...                                % idx       1
                       , traj_x, traj_y, traj_z ...             % idx     2-4
                       , traj_endeff_velocity ...               % idx     5-7
                       , traj_xdd, traj_ydd, traj_zdd ...       % idx    8-10
                       , traj_q0, traj_q1, traj_q2, traj_q3 ... % idx   11-14
                       , traj_ad,  traj_bd,  traj_gd ...        % idx   15-17
                       , traj_add, traj_bdd, traj_gdd ...       % idx   18-20
                       , traj_R_Fx, traj_R_Fy, traj_R_Fz ...    % idx   21-23 (Force of Force-Torque Sensor)
                       , traj_R_Tx, traj_R_Ty, traj_R_Tz ...    % idx   24-26 (Torque of Force-Torque Sensor)
                       , traj_R_LF_electrodes ...               % idx   27-45
                       , traj_R_LF_TDC ...                      % idx      46
                       , traj_R_LF_TAC ...                      % idx      47
                       , traj_R_LF_PDC ...                      % idx      48
                       , traj_R_LF_PACs ...                     % idx   49-70
                       , traj_R_LF_FX ...                       % idx      71 (BioTac Left-Finger Force-X)
                       , traj_R_LF_FY ...                       % idx      72 (BioTac Left-Finger Force-Y)
                       , traj_R_LF_FZ ...                       % idx      73 (BioTac Left-Finger Force-Z)
                       , traj_R_LF_MX ...                       % idx      74 (BioTac Left-Finger Torque-X)
                       , traj_R_LF_MY ...                       % idx      75 (BioTac Left-Finger Torque-Y)
                       , traj_R_LF_MZ ...                       % idx      76 (BioTac Left-Finger Torque-Z)
                       , traj_R_LF_POC_X ...                    % idx      77 (BioTac Left-Finger Point-of-Contact-X)
                       , traj_R_LF_POC_Y ...                    % idx      78 (BioTac Left-Finger Point-of-Contact-Y)
                       , traj_R_LF_POC_Z ...                    % idx      79 (BioTac Left-Finger Point-of-Contact-Z)
                       , traj_object_peg_board_xyz ...          % idx   80-82
                       , traj_object_peg_board_qwxyz ...        % idx   83-86
                       , traj_peg_tool_xyz ...                  % idx   87-89
                       , traj_peg_tool_qwxyz ...                % idx   90-93
                       , traj_R_RF_electrodes ...               % idx  94-112
                       , traj_R_RF_TDC ...                      % idx     113
                       , traj_R_RF_TAC ...                      % idx     114
                       , traj_R_RF_PDC ...                      % idx     115
                       , traj_R_RF_PACs ...                     % idx 116-137
                       , traj_R_RF_FX ...                       % idx     138 (BioTac Right-Finger Force-X)
                       , traj_R_RF_FY ...                       % idx     139 (BioTac Right-Finger Force-Y)
                       , traj_R_RF_FZ ...                       % idx     140 (BioTac Right-Finger Force-Z)
                       , traj_R_RF_MX ...                       % idx     141 (BioTac Right-Finger Torque-X)
                       , traj_R_RF_MY ...                       % idx     142 (BioTac Right-Finger Torque-Y)
                       , traj_R_RF_MZ ...                       % idx     143 (BioTac Right-Finger Torque-Z)
                       , traj_R_RF_POC_X ...                    % idx     144 (BioTac Right-Finger Point-of-Contact-X)
                       , traj_R_RF_POC_Y ...                    % idx     145 (BioTac Right-Finger Point-of-Contact-Y)
                       , traj_R_RF_POC_Z ...                    % idx     146 (BioTac Right-Finger Point-of-Contact-Z)
                       ];
            
%             figure;
%             plot(sm_traj(:,2:4));
%             title('before clipping');

            N_primitive     = size(retain_idx_cell, 2);
            
            % variable definition and initialization:
            R_base_to_object_peg_board             = eye(3);
            R_object_peg_board_to_base             = eye(3);
            T_object_peg_board_to_base             = eye(4);
            R_base_to_peg_tool_init     = eye(3);
            T_base_to_peg_tool_init     = eye(4);
            R_base_to_endeff_init           = eye(3);
            R_endeff_to_base_init           = eye(3);
            T_endeff_to_base_init           = eye(4);
            R_endeff_to_peg_tool_init   = eye(3);
            T_endeff_to_peg_tool_init   = eye(4);
            for np = 1:N_primitive
                out_iter_obj_prim_dir_path  = [out_iter_obj_path,'/prim',...
                                               num2str(np,'%02d'),'/'];
                if (~exist(out_iter_obj_prim_dir_path, 'dir'))
                    mkdir(out_iter_obj_prim_dir_path);
                end
                
                % segmented sensorimotor trajectory primitive:
                sm_primitive_traj       = sm_traj(retain_idx_cell{1,np},:);
                % offset the time so that it start with 0:
                sm_primitive_traj(:,1)  = sm_primitive_traj(:,1) - ...
                                          sm_primitive_traj(1,1);
                             
                if (is_performing_transformation_to_object_peg_board_coord_sys)
                    if (np == 1)
                        t_base_to_object_peg_board   = sm_primitive_traj(1,80:82)';
                        q_base_to_object_peg_board   = sm_primitive_traj(1,83:86)';
                        if ((norm(t_base_to_object_peg_board) == 0) || (abs(norm(q_base_to_object_peg_board) - 1) > 0.001))
                            fprintf('norm(t_base_to_object_peg_board)          = %f\n', norm(t_base_to_object_peg_board));
                            fprintf('norm(q_base_to_object_peg_board)          = %f\n', norm(q_base_to_object_peg_board));
                            error('object_peg_board is not detected properly.');
                        end

                        R_base_to_object_peg_board             = quaternion(q_base_to_object_peg_board).normalize.RotationMatrix;
                        T_base_to_object_peg_board             = eye(4);
                        T_base_to_object_peg_board(1:3,1:3)    = R_base_to_object_peg_board;
                        T_base_to_object_peg_board(1:3,4)      = t_base_to_object_peg_board;
                        R_object_peg_board_to_base             = R_base_to_object_peg_board';
                        T_object_peg_board_to_base             = eye(4);
                        t_object_peg_board_to_base             = -R_object_peg_board_to_base * t_base_to_object_peg_board;
                        T_object_peg_board_to_base(1:3,1:3)    = R_object_peg_board_to_base;
                        T_object_peg_board_to_base(1:3,4)      = t_object_peg_board_to_base;

                        t_base_to_peg_tool_init     = sm_primitive_traj(1,87:89)';
                        q_base_to_peg_tool_init     = sm_primitive_traj(1,90:93)';
                        if ((norm(t_base_to_peg_tool_init) == 0) || (abs(norm(q_base_to_peg_tool_init) - 1) > 0.001))
                            fprintf('norm(t_base_to_peg_tool_init)  = %f\n', norm(t_base_to_peg_tool_init));
                            fprintf('norm(q_base_to_peg_tool_init)  = %f\n', norm(q_base_to_peg_tool_init));
                            error('peg_tool is not initially detected properly.');
                        end

                        R_base_to_peg_tool_init         = quaternion(q_base_to_peg_tool_init).normalize.RotationMatrix;
                        T_base_to_peg_tool_init         = eye(4);
                        T_base_to_peg_tool_init(1:3,1:3)= R_base_to_peg_tool_init;
                        T_base_to_peg_tool_init(1:3,4)  = t_base_to_peg_tool_init;
%                         R_peg_tool_to_base_init         = R_base_to_peg_tool_init';
%                         T_peg_tool_to_base_init         = eye(4);
%                         T_peg_tool_to_base_init(1:3,1:3)= R_peg_tool_to_base_init;
%                         T_peg_tool_to_base_init(1:3,4)  = -R_peg_tool_to_base_init * t_base_to_peg_tool_init;

                        % compute transform pose (position and orientation)
                        % of peg_tool w.r.t. end effector coordinate system
                        % (assuming this transformation is constant for the whole demonstration)
                        t_base_to_endeff_init   = sm_primitive_traj(1, 2:4)';
                        q_base_to_endeff_init   = sm_primitive_traj(1, 11:14)';
                        R_base_to_endeff_init   = quaternion(q_base_to_endeff_init).normalize.RotationMatrix;
                        
                        T_endeff_to_base_init           = eye(4);
                        R_endeff_to_base_init           = R_base_to_endeff_init';
                        T_endeff_to_base_init(1:3,1:3)  = R_endeff_to_base_init;
                        T_endeff_to_base_init(1:3,4)    = -R_endeff_to_base_init * t_base_to_endeff_init;
                        
                        T_endeff_to_peg_tool_init   = T_endeff_to_base_init * T_base_to_peg_tool_init;
                        R_endeff_to_peg_tool_init   = T_endeff_to_peg_tool_init(1:3,1:3);
                        t_endeff_to_peg_tool_init   = T_endeff_to_peg_tool_init(1:3,4);
                        t_peg_tool_to_endeff_init   = -R_endeff_to_peg_tool_init' * t_endeff_to_peg_tool_init;
                        q_endeff_to_peg_tool_init   = quaternion.rotationmatrix(R_endeff_to_peg_tool_init).normalize.e';
                        R_endeff_to_peg_tool_init   = quaternion(q_endeff_to_peg_tool_init).normalize.RotationMatrix;
                        t_endeff_to_peg_tool_init   = -R_endeff_to_peg_tool_init * t_peg_tool_to_endeff_init;
                        R_peg_tool_to_endeff_init   = R_endeff_to_peg_tool_init';
                        T_endeff_to_peg_tool_init           = eye(4);
                        T_endeff_to_peg_tool_init(1:3,1:3)  = R_endeff_to_peg_tool_init;
                        T_endeff_to_peg_tool_init(1:3,4)    = t_endeff_to_peg_tool_init;
                        T_peg_tool_to_endeff_init           = eye(4);
                        T_peg_tool_to_endeff_init(1:3,1:3)  = R_peg_tool_to_endeff_init;
                        T_peg_tool_to_endeff_init(1:3,4)    = t_peg_tool_to_endeff_init;
                        fprintf('The (assumed) constant transformation from endeff to peg_tool is computed.\n');
                    end
                    
                    for p_idx=1:size(sm_primitive_traj,1)
                        % compute transform pose (position and orientation)
                        % of peg_tool w.r.t. object_peg_board coordinate system
                        t_base_to_endeff  = sm_primitive_traj(p_idx, 2:4)';
                        q_base_to_endeff  = sm_primitive_traj(p_idx, 11:14)';
                        R_base_to_endeff  = quaternion(q_base_to_endeff).normalize.RotationMatrix;
                        
                        T_base_to_endeff            = eye(4);
                        T_base_to_endeff(1:3,1:3)   = R_base_to_endeff;
                        T_base_to_endeff(1:3,4)     = t_base_to_endeff;
                        
                        T_object_peg_board_to_endeff           = T_object_peg_board_to_base * T_base_to_endeff;
                        
                        % log the results for pose (position and
                        % orientation)
                        if (is_peg_tool_tip_of_transform_chain)
                            T_object_peg_board_to_peg_tool     = T_object_peg_board_to_endeff * T_endeff_to_peg_tool_init;
                            R_object_peg_board_to_peg_tool     = T_object_peg_board_to_peg_tool(1:3,1:3);
                            t_object_peg_board_to_peg_tool     = T_object_peg_board_to_peg_tool(1:3,4);
                            t_peg_tool_to_object_peg_board     = -R_object_peg_board_to_peg_tool' * t_object_peg_board_to_peg_tool;
                            q_object_peg_board_to_peg_tool     = quaternion.rotationmatrix(R_object_peg_board_to_peg_tool).normalize.e';
                            R_object_peg_board_to_peg_tool     = quaternion(q_object_peg_board_to_peg_tool).normalize.RotationMatrix;
                            t_object_peg_board_to_peg_tool     = -R_object_peg_board_to_peg_tool * t_peg_tool_to_object_peg_board;
                            T_object_peg_board_to_peg_tool         = eye(4);
                            T_object_peg_board_to_peg_tool(1:3,1:3)= R_object_peg_board_to_peg_tool;
                            T_object_peg_board_to_peg_tool(1:3,4)  = t_object_peg_board_to_peg_tool;
                        
                            sm_primitive_traj(p_idx, 2:4)   = t_object_peg_board_to_peg_tool';
                            sm_primitive_traj(p_idx, 11:14) = q_object_peg_board_to_peg_tool';
                        else
                            t_object_peg_board_to_endeff           = T_object_peg_board_to_endeff(1:3,4);
                            R_object_peg_board_to_endeff           = T_object_peg_board_to_endeff(1:3,1:3);
                            q_object_peg_board_to_endeff           = quaternion.rotationmatrix(R_object_peg_board_to_endeff).normalize.e';
                        
                            sm_primitive_traj(p_idx, 2:4)   = t_object_peg_board_to_endeff';
                            sm_primitive_traj(p_idx, 11:14) = q_object_peg_board_to_endeff';
                        end
                        
                        % transform vectors (velocity, acceleration,
                        % angular velocity, and angular acceleration)
                        % of peg_tool w.r.t. object_peg_board coordinate system
                        v_base_to_endeff                = sm_primitive_traj(p_idx, 5:7)';
                        a_base_to_endeff                = sm_primitive_traj(p_idx, 8:10)';
                        
                        omega_base_to_endeff            = sm_primitive_traj(p_idx, 15:17)';
                        alpha_base_to_endeff            = sm_primitive_traj(p_idx, 18:20)';
                        
                        if (is_peg_tool_tip_of_transform_chain)
                            v_base_to_peg_tool          = v_base_to_endeff + (computeCrossProductMatrix(omega_base_to_endeff) * R_base_to_endeff * t_endeff_to_peg_tool_init);
                            omega_base_to_peg_tool      = omega_base_to_endeff;
                            v_object_peg_board_to_peg_tool     = R_object_peg_board_to_base * v_base_to_peg_tool;
                            omega_object_peg_board_to_peg_tool = R_object_peg_board_to_base * omega_base_to_peg_tool;

                            a_base_to_peg_tool          = a_base_to_endeff + (computeCrossProductMatrix(alpha_base_to_endeff) * R_base_to_endeff * t_endeff_to_peg_tool_init);
                            alpha_base_to_peg_tool      = alpha_base_to_endeff;
                            a_object_peg_board_to_peg_tool     = R_object_peg_board_to_base * a_base_to_peg_tool;
                            alpha_object_peg_board_to_peg_tool = R_object_peg_board_to_base * alpha_base_to_peg_tool;

                            % log the results for vectors (velocity, acceleration,
                            % angular velocity, and angular acceleration)
                            sm_primitive_traj(p_idx, 5:7)   = v_object_peg_board_to_peg_tool';
                            sm_primitive_traj(p_idx, 8:10)  = a_object_peg_board_to_peg_tool';
                            sm_primitive_traj(p_idx, 15:17) = omega_object_peg_board_to_peg_tool';
                            sm_primitive_traj(p_idx, 18:20) = alpha_object_peg_board_to_peg_tool';
                        else
                            v_object_peg_board_to_endeff           = R_object_peg_board_to_base * v_base_to_endeff;
                            a_object_peg_board_to_endeff           = R_object_peg_board_to_base * a_base_to_endeff;
                            omega_object_peg_board_to_endeff       = R_object_peg_board_to_base * omega_base_to_endeff;
                            alpha_object_peg_board_to_endeff       = R_object_peg_board_to_base * alpha_base_to_endeff;
                            
                            sm_primitive_traj(p_idx, 5:7)   = v_object_peg_board_to_endeff';
                            sm_primitive_traj(p_idx, 8:10)  = a_object_peg_board_to_endeff';
                            sm_primitive_traj(p_idx, 15:17) = omega_object_peg_board_to_endeff';
                            sm_primitive_traj(p_idx, 18:20) = alpha_object_peg_board_to_endeff';
                        end
                    end
                end
                
%                 for dim_idx=0:2
%                     sm_primitive_traj(:, 5+dim_idx) = diffnc(sm_primitive_traj(:, 2+dim_idx), 1.0/freq);
%                     sm_primitive_traj(:, 8+dim_idx) = diffnc(sm_primitive_traj(:, 5+dim_idx), 1.0/freq);
%                 end
                
%                 figure;
%                 plot(sm_primitive_traj(:,2:4)); % plot Cartesian position traj
%                 title(['segmented sensorimotor primitive #', ...
%                        num2str(np,'%02d'), ...
%                        ' Cartesian position trajectory #', ...
%                        num2str(data_file_count,'%02d')]);
%                 keyboard;

%                 figure;
%                 plot(sm_primitive_traj(:,21:23)); % plot Force of Force-Torque Sensor traj
%                 title(['segmented sensorimotor primitive #', ...
%                        num2str(np,'%02d'), ...
%                        ' Force of Force-Torque Sensor trajectory #', ...
%                        num2str(data_file_count,'%02d')]);
%                 keyboard;

                out_data_file_path = [out_iter_obj_prim_dir_path,'/',...
                                      num2str(data_file_count,'%02d'),'.txt'];
                dlmwrite(out_data_file_path, sm_primitive_traj, ...
                         'delimiter', ' ');
            end
                
            data_file_count = data_file_count + 1;
        end
    end
end