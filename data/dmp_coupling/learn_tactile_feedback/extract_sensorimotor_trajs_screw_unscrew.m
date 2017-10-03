% Author: Giovanni Sutanto
% Date  : Jan 30, 2017

close all;
clear all;
clc;

addpath('../../../matlab/utilities/');
addpath('../../../matlab/utilities/clmcplot/');
addpath('../../../matlab/utilities/quaternion/');

% Change in_data_dir_path to the path of the input data directory, as
% necessary:
in_data_dir_path    = '~/Desktop/dmp_demos/screw_unscrew/';
out_data_dir_path   = [pwd, '/'];

is_performing_transformation_to_truck_toy_coord_sys     = 1;
is_tool_adaptor_tip_of_transform_chain                  = 0;

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
            [traj_truck_toy_xyz]        = clmcplot_getvariables(D, vars, {'truck_toy_x','truck_toy_y','truck_toy_z'});
            [traj_truck_toy_qwxyz]      = clmcplot_getvariables(D, vars, {'truck_toy_qw','truck_toy_qx','truck_toy_qy','truck_toy_qz'});
            [traj_tool_adaptor_xyz]     = clmcplot_getvariables(D, vars, {'tool_adaptor_x','tool_adaptor_y','tool_adaptor_z'});
            [traj_tool_adaptor_qwxyz]   = clmcplot_getvariables(D, vars, {'tool_adaptor_qw','tool_adaptor_qx','tool_adaptor_qy','tool_adaptor_qz'});
            % dummy tool_adaptor orientation for robustness testing:
%             [traj_tool_adaptor_qwxyz]   = repmat([cos(pi/3), 0, sin(pi/3), 0],size(time,1),1);
            
            traj_endeff_velocity    = [traj_xd, traj_yd, traj_zd];
                   
            [ retain_idx_cell ] = getDataClippingRetainIndex( ...
                                        traj_endeff_velocity, 0 );
            
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
            
            sm_traj = [time  ...                                % idx     1
                       , traj_x, traj_y, traj_z ...             % idx   2-4
                       , traj_endeff_velocity ...               % idx   5-7
                       , traj_xdd, traj_ydd, traj_zdd ...       % idx  8-10
                       , traj_q0, traj_q1, traj_q2, traj_q3 ... % idx 11-14
                       , traj_ad,  traj_bd,  traj_gd ...        % idx 15-17
                       , traj_add, traj_bdd, traj_gdd ...       % idx 18-20
                       , traj_R_Fx, traj_R_Fy, traj_R_Fz ...    % idx 21-23 (Force of Force-Torque Sensor)
                       , traj_R_Tx, traj_R_Ty, traj_R_Tz ...    % idx 24-26 (Torque of Force-Torque Sensor)
                       , traj_R_LF_electrodes ...               % idx 27-45
                       , traj_R_LF_TDC ...                      % idx    46
                       , traj_R_LF_TAC ...                      % idx    47
                       , traj_R_LF_PDC ...                      % idx    48
                       , traj_R_LF_PACs ...                     % idx 49-70
                       , traj_R_LF_FX ...                       % idx    71 (BioTac Force-X)
                       , traj_R_LF_FY ...                       % idx    72 (BioTac Force-Y)
                       , traj_R_LF_FZ ...                       % idx    73 (BioTac Force-Z)
                       , traj_R_LF_MX ...                       % idx    74 (BioTac Torque-X)
                       , traj_R_LF_MY ...                       % idx    75 (BioTac Torque-Y)
                       , traj_R_LF_MZ ...                       % idx    76 (BioTac Torque-Z)
                       , traj_R_LF_POC_X ...                    % idx    77 (Point-of-Contact-X)
                       , traj_R_LF_POC_Y ...                    % idx    78 (Point-of-Contact-Y)
                       , traj_R_LF_POC_Z ...                    % idx    79 (Point-of-Contact-Z)
                       , traj_truck_toy_xyz ...                 % idx 80-82
                       , traj_truck_toy_qwxyz ...               % idx 83-86
                       , traj_tool_adaptor_xyz ...              % idx 87-89
                       , traj_tool_adaptor_qwxyz ...            % idx 90-93
                       ];
            
%             figure;
%             plot(sm_traj(:,2:4));
%             title('before clipping');

            N_primitive     = size(retain_idx_cell, 2);
            
            % variable definition and initialization:
            R_base_to_truck_toy             = eye(3);
            R_truck_toy_to_base             = eye(3);
            T_truck_toy_to_base             = eye(4);
            R_base_to_tool_adaptor_init     = eye(3);
            T_base_to_tool_adaptor_init     = eye(4);
            R_base_to_endeff_init           = eye(3);
            R_endeff_to_base_init           = eye(3);
            T_endeff_to_base_init           = eye(4);
            R_endeff_to_tool_adaptor_init   = eye(3);
            T_endeff_to_tool_adaptor_init   = eye(4);
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
                             
                if (is_performing_transformation_to_truck_toy_coord_sys)
                    if (np == 1)
                        t_base_to_truck_toy   = sm_primitive_traj(1,80:82)';
                        q_base_to_truck_toy   = sm_primitive_traj(1,83:86)';
                        if ((norm(t_base_to_truck_toy) == 0) || (abs(norm(q_base_to_truck_toy) - 1) > 0.001))
                            fprintf('norm(t_base_to_truck_toy)          = %f\n', norm(t_base_to_truck_toy));
                            fprintf('norm(q_base_to_truck_toy)          = %f\n', norm(q_base_to_truck_toy));
                            error('truck_toy is not detected properly.');
                        end

                        R_base_to_truck_toy             = quaternion(q_base_to_truck_toy).normalize.RotationMatrix;
                        T_base_to_truck_toy             = eye(4);
                        T_base_to_truck_toy(1:3,1:3)    = R_base_to_truck_toy;
                        T_base_to_truck_toy(1:3,4)      = t_base_to_truck_toy;
                        R_truck_toy_to_base             = R_base_to_truck_toy';
                        T_truck_toy_to_base             = eye(4);
                        t_truck_toy_to_base             = -R_truck_toy_to_base * t_base_to_truck_toy;
                        T_truck_toy_to_base(1:3,1:3)    = R_truck_toy_to_base;
                        T_truck_toy_to_base(1:3,4)      = t_truck_toy_to_base;

                        t_base_to_tool_adaptor_init     = sm_primitive_traj(1,87:89)';
                        q_base_to_tool_adaptor_init     = sm_primitive_traj(1,90:93)';
                        if ((norm(t_base_to_tool_adaptor_init) == 0) || (abs(norm(q_base_to_tool_adaptor_init) - 1) > 0.001))
                            fprintf('norm(t_base_to_tool_adaptor_init)  = %f\n', norm(t_base_to_tool_adaptor_init));
                            fprintf('norm(q_base_to_tool_adaptor_init)  = %f\n', norm(q_base_to_tool_adaptor_init));
                            error('tool_adaptor is not initially detected properly.');
                        end

                        R_base_to_tool_adaptor_init         = quaternion(q_base_to_tool_adaptor_init).normalize.RotationMatrix;
                        T_base_to_tool_adaptor_init         = eye(4);
                        T_base_to_tool_adaptor_init(1:3,1:3)= R_base_to_tool_adaptor_init;
                        T_base_to_tool_adaptor_init(1:3,4)  = t_base_to_tool_adaptor_init;
%                         R_tool_adaptor_to_base_init         = R_base_to_tool_adaptor_init';
%                         T_tool_adaptor_to_base_init         = eye(4);
%                         T_tool_adaptor_to_base_init(1:3,1:3)= R_tool_adaptor_to_base_init;
%                         T_tool_adaptor_to_base_init(1:3,4)  = -R_tool_adaptor_to_base_init * t_base_to_tool_adaptor_init;

                        % compute transform pose (position and orientation)
                        % of tool_adaptor w.r.t. end effector coordinate system
                        % (assuming this transformation is constant for the whole demonstration)
                        t_base_to_endeff_init   = sm_primitive_traj(1, 2:4)';
                        q_base_to_endeff_init   = sm_primitive_traj(1, 11:14)';
                        R_base_to_endeff_init   = quaternion(q_base_to_endeff_init).normalize.RotationMatrix;
                        
                        T_endeff_to_base_init           = eye(4);
                        R_endeff_to_base_init           = R_base_to_endeff_init';
                        T_endeff_to_base_init(1:3,1:3)  = R_endeff_to_base_init;
                        T_endeff_to_base_init(1:3,4)    = -R_endeff_to_base_init * t_base_to_endeff_init;
                        
                        T_endeff_to_tool_adaptor_init   = T_endeff_to_base_init * T_base_to_tool_adaptor_init;
                        R_endeff_to_tool_adaptor_init   = T_endeff_to_tool_adaptor_init(1:3,1:3);
                        t_endeff_to_tool_adaptor_init   = T_endeff_to_tool_adaptor_init(1:3,4);
                        t_tool_adaptor_to_endeff_init   = -R_endeff_to_tool_adaptor_init' * t_endeff_to_tool_adaptor_init;
                        q_endeff_to_tool_adaptor_init   = quaternion.rotationmatrix(R_endeff_to_tool_adaptor_init).normalize.e';
                        R_endeff_to_tool_adaptor_init   = quaternion(q_endeff_to_tool_adaptor_init).normalize.RotationMatrix;
                        t_endeff_to_tool_adaptor_init   = -R_endeff_to_tool_adaptor_init * t_tool_adaptor_to_endeff_init;
                        R_tool_adaptor_to_endeff_init   = R_endeff_to_tool_adaptor_init';
                        T_endeff_to_tool_adaptor_init           = eye(4);
                        T_endeff_to_tool_adaptor_init(1:3,1:3)  = R_endeff_to_tool_adaptor_init;
                        T_endeff_to_tool_adaptor_init(1:3,4)    = t_endeff_to_tool_adaptor_init;
                        T_tool_adaptor_to_endeff_init           = eye(4);
                        T_tool_adaptor_to_endeff_init(1:3,1:3)  = R_tool_adaptor_to_endeff_init;
                        T_tool_adaptor_to_endeff_init(1:3,4)    = t_tool_adaptor_to_endeff_init;
                        fprintf('The (assumed) constant transformation from endeff to tool_adaptor is computed.\n');
                    end
                    
                    for p_idx=1:size(sm_primitive_traj,1)
                        % compute transform pose (position and orientation)
                        % of tool_adaptor w.r.t. truck_toy coordinate system
                        t_base_to_endeff  = sm_primitive_traj(p_idx, 2:4)';
                        q_base_to_endeff  = sm_primitive_traj(p_idx, 11:14)';
                        R_base_to_endeff  = quaternion(q_base_to_endeff).normalize.RotationMatrix;
                        
                        T_base_to_endeff            = eye(4);
                        T_base_to_endeff(1:3,1:3)   = R_base_to_endeff;
                        T_base_to_endeff(1:3,4)     = t_base_to_endeff;
                        
                        T_truck_toy_to_endeff           = T_truck_toy_to_base * T_base_to_endeff;
                        
                        % log the results for pose (position and
                        % orientation)
                        if (is_tool_adaptor_tip_of_transform_chain)
                            T_truck_toy_to_tool_adaptor     = T_truck_toy_to_endeff * T_endeff_to_tool_adaptor_init;
                            R_truck_toy_to_tool_adaptor     = T_truck_toy_to_tool_adaptor(1:3,1:3);
                            t_truck_toy_to_tool_adaptor     = T_truck_toy_to_tool_adaptor(1:3,4);
                            t_tool_adaptor_to_truck_toy     = -R_truck_toy_to_tool_adaptor' * t_truck_toy_to_tool_adaptor;
                            q_truck_toy_to_tool_adaptor     = quaternion.rotationmatrix(R_truck_toy_to_tool_adaptor).normalize.e';
                            R_truck_toy_to_tool_adaptor     = quaternion(q_truck_toy_to_tool_adaptor).normalize.RotationMatrix;
                            t_truck_toy_to_tool_adaptor     = -R_truck_toy_to_tool_adaptor * t_tool_adaptor_to_truck_toy;
                            T_truck_toy_to_tool_adaptor         = eye(4);
                            T_truck_toy_to_tool_adaptor(1:3,1:3)= R_truck_toy_to_tool_adaptor;
                            T_truck_toy_to_tool_adaptor(1:3,4)  = t_truck_toy_to_tool_adaptor;
                        
                            sm_primitive_traj(p_idx, 2:4)   = t_truck_toy_to_tool_adaptor';
                            sm_primitive_traj(p_idx, 11:14) = q_truck_toy_to_tool_adaptor';
                        else
                            t_truck_toy_to_endeff           = T_truck_toy_to_endeff(1:3,4);
                            R_truck_toy_to_endeff           = T_truck_toy_to_endeff(1:3,1:3);
                            q_truck_toy_to_endeff           = quaternion.rotationmatrix(R_truck_toy_to_endeff).normalize.e';
                        
                            sm_primitive_traj(p_idx, 2:4)   = t_truck_toy_to_endeff';
                            sm_primitive_traj(p_idx, 11:14) = q_truck_toy_to_endeff';
                        end
                        
                        % transform vectors (velocity, acceleration,
                        % angular velocity, and angular acceleration)
                        % of tool_adaptor w.r.t. truck_toy coordinate system
                        v_base_to_endeff                = sm_primitive_traj(p_idx, 5:7)';
                        a_base_to_endeff                = sm_primitive_traj(p_idx, 8:10)';
                        
                        omega_base_to_endeff            = sm_primitive_traj(p_idx, 15:17)';
                        alpha_base_to_endeff            = sm_primitive_traj(p_idx, 18:20)';
                        
                        if (is_tool_adaptor_tip_of_transform_chain)
                            v_base_to_tool_adaptor          = v_base_to_endeff + (computeCrossProductMatrix(omega_base_to_endeff) * R_base_to_endeff * t_endeff_to_tool_adaptor_init);
                            omega_base_to_tool_adaptor      = omega_base_to_endeff;
                            v_truck_toy_to_tool_adaptor     = R_truck_toy_to_base * v_base_to_tool_adaptor;
                            omega_truck_toy_to_tool_adaptor = R_truck_toy_to_base * omega_base_to_tool_adaptor;

                            a_base_to_tool_adaptor          = a_base_to_endeff + (computeCrossProductMatrix(alpha_base_to_endeff) * R_base_to_endeff * t_endeff_to_tool_adaptor_init);
                            alpha_base_to_tool_adaptor      = alpha_base_to_endeff;
                            a_truck_toy_to_tool_adaptor     = R_truck_toy_to_base * a_base_to_tool_adaptor;
                            alpha_truck_toy_to_tool_adaptor = R_truck_toy_to_base * alpha_base_to_tool_adaptor;

                            % log the results for vectors (velocity, acceleration,
                            % angular velocity, and angular acceleration)
                            sm_primitive_traj(p_idx, 5:7)   = v_truck_toy_to_tool_adaptor';
                            sm_primitive_traj(p_idx, 8:10)  = a_truck_toy_to_tool_adaptor';
                            sm_primitive_traj(p_idx, 15:17) = omega_truck_toy_to_tool_adaptor';
                            sm_primitive_traj(p_idx, 18:20) = alpha_truck_toy_to_tool_adaptor';
                        else
                            v_truck_toy_to_endeff           = R_truck_toy_to_base * v_base_to_endeff;
                            a_truck_toy_to_endeff           = R_truck_toy_to_base * a_base_to_endeff;
                            omega_truck_toy_to_endeff       = R_truck_toy_to_base * omega_base_to_endeff;
                            alpha_truck_toy_to_endeff       = R_truck_toy_to_base * alpha_base_to_endeff;
                            
                            sm_primitive_traj(p_idx, 5:7)   = v_truck_toy_to_endeff';
                            sm_primitive_traj(p_idx, 8:10)  = a_truck_toy_to_endeff';
                            sm_primitive_traj(p_idx, 15:17) = omega_truck_toy_to_endeff';
                            sm_primitive_traj(p_idx, 18:20) = alpha_truck_toy_to_endeff';
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