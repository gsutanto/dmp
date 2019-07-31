% Author: Giovanni Sutanto
% Date  : July 2017

close all;
clear all;
clc;

addpath('../../../matlab/utilities/');
addpath('../../../matlab/utilities/clmcplot/');
addpath('../../../matlab/utilities/quaternion/');

specific_task_type 	= 'scraping_w_tool';

% Change in_data_dir_path to the path of the input data directory, as
% necessary:
in_data_dir_path    = ['~/Desktop/dmp_demos/',specific_task_type,'/'];
out_data_dir_path   = [pwd, '/../../../../amd_clmc_dmp_data/dmp_coupling/learn_tactile_feedback/',specific_task_type,'/'];

is_baseline_first_to_be_processed   = 1;

N_primitive                         = 3;
max_num_coupled_settings_considered = 16;

contents    = dir(in_data_dir_path);

if (is_baseline_first_to_be_processed)
    % swap folder list order, such that baseline is seen first before the
    % non-baselines:
    for content_idx = 1:size(contents, 1)
        if (strcmp(contents(content_idx,1).name, '1') == 1)
            content_idx_to_be_swapped   = content_idx - 1;
            content_to_be_swapped       = contents(content_idx_to_be_swapped,1);
        end
        if (strcmp(contents(content_idx,1).name, 'baseline') == 1)
            baseline_content_idx        = content_idx;
            baseline_content            = contents(baseline_content_idx,1);
        end
    end
    contents(content_idx_to_be_swapped,1)   = baseline_content;
    contents(baseline_content_idx,1)        = content_to_be_swapped;
end

for iter_obj = contents'
    in_iter_obj_path    = [in_data_dir_path, iter_obj.name];
    if ((strcmp(iter_obj.name, '.') == 0) && (strcmp(iter_obj.name, 'human_baseline') == 0)...
        && (strcmp(iter_obj.name, 'old') == 0) && (strcmp(iter_obj.name, '..') == 0)...
        && (isdir(in_iter_obj_path)))
    
        % if not the baseline demonstrations ...
        if (strcmp(iter_obj.name, 'baseline') == 0)
            setting_number  = str2num(iter_obj.name);
            % if setting_number is greater than
            % max_num_coupled_settings_considered ...
            if (setting_number > max_num_coupled_settings_considered)
                % skip it.
                continue;
%             elseif (setting_number ~= 2)
%                 % skip it.
%                 continue;
            end
        end
        
        out_iter_obj_path   = [out_data_dir_path, iter_obj.name];
        recreateDir(out_iter_obj_path);
        data_files      = dir(strcat(in_iter_obj_path,'/','d*'));
        data_file_count = 1;
        for data_file = data_files'
            in_data_file_path  = [in_iter_obj_path,'/',data_file.name];
            fprintf('Processing %s...\n', in_data_file_path);
            [D,vars,freq]   = clmcplot_convert(in_data_file_path);

            time    = clmcplot_getvariables(D, vars, {'time'});
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
            [traj_R_Fx_global, traj_R_Fy_global, traj_R_Fz_global] = clmcplot_getvariables(D, vars, {'R_HAND_d_FX','R_HAND_d_FY','R_HAND_d_FZ'});
            [traj_R_Tx_global, traj_R_Ty_global, traj_R_Tz_global] = clmcplot_getvariables(D, vars, {'R_HAND_d_TX','R_HAND_d_TY','R_HAND_d_TZ'});
            [traj_R_Fz_gate_Pgain, traj_R_Fz_gate_Igain] = clmcplot_getvariables(D, vars, {'R_HAND_F_posZ_Pgain','R_HAND_F_posZ_Igain'});
            [traj_R_Ty_gate_Pgain, traj_R_Ty_gate_Igain] = clmcplot_getvariables(D, vars, {'R_HAND_F_rotY_Pgain','R_HAND_F_rotY_Igain'});
            traj_R_LF_electrodes    = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_LF_','E',[1:19]));
            traj_R_LF_TDC   = clmcplot_getvariables(D, vars, {'R_LF_TDC'});
            traj_R_LF_TAC   = clmcplot_getvariables(D, vars, {'R_LF_TAC'});
            traj_R_LF_PDC   = clmcplot_getvariables(D, vars, {'R_LF_PDC'});
            traj_R_LF_PACs  = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_LF_','PAC_',[1:22]));
            [traj_R_LF_FX, traj_R_LF_FY, traj_R_LF_FZ]  = clmcplot_getvariables(D, vars, {'R_LF_FX', 'R_LF_FY', 'R_LF_FZ'});
            [traj_R_LF_MX, traj_R_LF_MY, traj_R_LF_MZ]  = clmcplot_getvariables(D, vars, {'R_LF_MX', 'R_LF_MY', 'R_LF_MZ'});
            [traj_R_LF_POC_X, traj_R_LF_POC_Y, traj_R_LF_POC_Z]  = clmcplot_getvariables(D, vars, {'R_LF_POC_X', 'R_LF_POC_Y', 'R_LF_POC_Z'});
            [traj_scraping_board_xyz]   = zeros(size(time, 1), 3);  % unused data
            [traj_scraping_board_qwxyz] = zeros(size(time, 1), 4);  % unused data
            [traj_tool_adaptor_xyz]     = zeros(size(time, 1), 3);  % unused data
            [traj_tool_adaptor_qwxyz]   = zeros(size(time, 1), 4);  % unused data
            traj_R_RF_electrodes    = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_RF_','E',[1:19]));
            traj_R_RF_TDC   = clmcplot_getvariables(D, vars, {'R_RF_TDC'});
            traj_R_RF_TAC   = clmcplot_getvariables(D, vars, {'R_RF_TAC'});
            traj_R_RF_PDC   = clmcplot_getvariables(D, vars, {'R_RF_PDC'});
            traj_R_RF_PACs  = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('R_RF_','PAC_',[1:22]));
            [traj_R_RF_FX, traj_R_RF_FY, traj_R_RF_FZ]  = clmcplot_getvariables(D, vars, {'R_RF_FX', 'R_RF_FY', 'R_RF_FZ'});
            [traj_R_RF_MX, traj_R_RF_MY, traj_R_RF_MZ]  = clmcplot_getvariables(D, vars, {'R_RF_MX', 'R_RF_MY', 'R_RF_MZ'});
            [traj_R_RF_POC_X, traj_R_RF_POC_Y, traj_R_RF_POC_Z]  = clmcplot_getvariables(D, vars, {'R_RF_POC_X', 'R_RF_POC_Y', 'R_RF_POC_Z'});
            traj_joint_positions    = clmcplot_getvariables(D, vars, ...
                                                            getDataNamesFromCombinations('R_', ...
                                                                {'SFE_', 'SAA_', 'HR_', 'EB_', 'WR_', 'WFE_', 'WAA_'}, ...
                                                                {'th'}));
        	current_prim_no = clmcplot_getvariables(D, vars, {'current_prim_no'});
            ul_curr_prim_no = clmcplot_getvariables(D, vars, {'ul_curr_prim_no'});
            
            dt  = 1.0/freq;
            
            %% Store the Initial Joint States (especially Posture)
             % for Later Deployment on the Real Robot
            
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
            
            %% Stack the Data together into a Huge Matrix
            
            sm_traj = [time  ...                                % idx       1
                       , traj_x, traj_y, traj_z ...             % idx     2-4
                       , traj_xd, traj_yd, traj_zd ...          % idx     5-7
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
                       , traj_R_Fx_global, traj_R_Fy_global, traj_R_Fz_global ...   % idx   80-82 (Force of Force-Torque Sensor, world/global coord. system; previously this space was used for traj_scraping_board_xyz)
                       , traj_R_Fz_gate_Pgain, traj_R_Fz_gate_Igain, traj_R_Ty_gate_Pgain, traj_R_Ty_gate_Igain ...	% idx   83-86 (PI gating gains for z-axis Force Control and y-axis Torque Control based on Force-Torque Sensor; previously this space was used for traj_scraping_board_qwxyz)
                       , traj_R_Tx_global, traj_R_Ty_global, traj_R_Tz_global ...	% idx   87-89 (Torque of Force-Torque Sensor, world/global coord. system; previously this space was used for traj_tool_adaptor_xyz)
                       , traj_tool_adaptor_qwxyz ...            % idx   90-93
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
                       , traj_joint_positions ...               % idx 147-153 (Joint Positions/Coordinates)
                       ];
            
            %% Segmentation of the Primitives
            
            if (size(find(current_prim_no ~= -1),1) == 0)
                valid_current_prim_no_idx       = [1; find(time > 0.0)];
                current_prim_no                 = ul_curr_prim_no(valid_current_prim_no_idx(1):valid_current_prim_no_idx(end),:);
            elseif (size(find(current_prim_no ~= -1),1) > 0)
                valid_current_prim_no_idx       = find(current_prim_no == -1);
                current_prim_no                 = current_prim_no(valid_current_prim_no_idx(1):valid_current_prim_no_idx(end),:);
            end
            
            for np = 1:N_primitive
                out_iter_obj_prim_dir_path  = [out_iter_obj_path,'/prim',...
                                               num2str(np,'%02d'),'/'];
                createDirIfNotExist(out_iter_obj_prim_dir_path);
                
                prim_data_indices           = find(current_prim_no == (np-1));
                start_clipping_index        = prim_data_indices(1) + 1;     % plus one because of the nature of the C++ program that recorded this...
                end_clipping_index          = prim_data_indices(end) + 1;   % plus one because of the nature of the C++ program that recorded this...
                
                % segmented sensorimotor trajectory primitive:
                sm_primitive_traj           = sm_traj(start_clipping_index:end_clipping_index,:);
                % offset the time so that it start with 0:
                sm_primitive_traj(:,1)      = sm_primitive_traj(:,1) - ...
                                              sm_primitive_traj(1,1);

                out_data_file_path = [out_iter_obj_prim_dir_path,'/',...
                                      num2str(data_file_count,'%02d'),'.txt'];
                dlmwrite(out_data_file_path, sm_primitive_traj, ...
                         'delimiter', ' ');
            end
            
            data_file_count = data_file_count + 1;
        end
    end
end