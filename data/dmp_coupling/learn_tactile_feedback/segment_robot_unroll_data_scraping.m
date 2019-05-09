% Author: Giovanni Sutanto
% Date  : July 4, 2017

close all;
clear all;
clc;

addpath('../../../matlab/utilities/');
addpath('../../../matlab/utilities/clmcplot/');
addpath('../../../matlab/utilities/quaternion/');

generic_task_type           = 'scraping';
specific_task_type          = 'scraping_w_tool';
% date                        = '20170831_3';
% additional_description      = '_after_pruning_inconsistent_demos_positive_side';
date                        = '20190508';
additional_description      = '';
experiment_name             = [date,'_',specific_task_type,'_correctable',additional_description];

in_data_root_dir_path       = ['~/Desktop/dmp_robot_unroll_results/',generic_task_type,'/',experiment_name,'/robot/'];
out_data_root_dir_path      = [pwd, '/',generic_task_type,'/robot_unroll_data/'];
sample_data_root_dir_path   = [pwd, '/',generic_task_type,'/unroll_test_dataset/all_prims/pos/mean/'];

if (~exist(in_data_root_dir_path, 'dir'))
    error([in_data_root_dir_path, ' does NOT exist!']);
end

unroll_types                = {'b', 'c'};   % 'b' = baseline; 'c' = coupled

N_prims                     = 3;

sample_traj_cell            = cell(N_prims, 1);

%% Dictionary Definition/Mapping

dict            = containers.Map;
dict('p2_5')    = 1;
dict('p5')      = 2;
dict('p7_5')    = 3;
dict('p10')     = 4;
dict('n2_5')    = 5;
dict('n5')      = 6;
dict('n7_5')    = 7;
dict('n10')     = 8;
dict('n12_5')   = 9;
dict('zero')    = 10;

% end of Dictionary Definition/Mapping

%% Enumeration of Settings Tested/Unrolled by the Robot

in_data_root_dir_path_contents  = dir(in_data_root_dir_path);

in_data_dir_names               = cell(0);
out_data_dir_names              = cell(0);
iter_obj_count  = 1;
for iter_obj = in_data_root_dir_path_contents'
    if ((strcmp(iter_obj.name, '.') == 0) && (strcmp(iter_obj.name, '..') == 0) && ...
        (isKey(dict, iter_obj.name) == 1))
        in_data_dir_names{1,iter_obj_count}     = iter_obj.name;
        out_data_dir_names{1,iter_obj_count}    = num2str(dict(iter_obj.name));
        iter_obj_count                          = iter_obj_count + 1;
    end
end

N_settings_considered           = size(in_data_dir_names, 2);

% end of Enumeration of Settings Tested/Unrolled by the Robot

%% Load Sample Trajectory Primitives' Data
for np = 1:N_prims
    sample_file_path        = [sample_data_root_dir_path, num2str(np), '.txt'];
    
    sample_traj_cell{np, 1} = dlmread(sample_file_path);
end

for nut = 1:size(unroll_types, 2)
    out_data_unroll_type_root_dir_path  = [out_data_root_dir_path, unroll_types{1, nut}, '/'];
    recreateDir(out_data_unroll_type_root_dir_path);
    for ns = 1:N_settings_considered
        %% Paths Determination
        in_data_dir_path    = [in_data_root_dir_path, in_data_dir_names{1,ns}, '/', unroll_types{1, nut}, '/'];
        out_data_dir_path   = [out_data_unroll_type_root_dir_path, out_data_dir_names{1,ns}, '/'];
        recreateDir(out_data_dir_path);

        %% Data Loading
        data_files              = dir([in_data_dir_path,'/d*']);
        data_file_count         = 1;
        for data_file = data_files'
            in_data_file_path   = [in_data_dir_path,'/',data_file.name];
            fprintf('Processing %s...\n', in_data_file_path);
            [D,vars,freq]       = clmcplot_convert(in_data_file_path);
            dt                  = 1.0/freq;

            time    = clmcplot_getvariables(D, vars, {'time'});
            [traj_x, traj_y, traj_z]        = clmcplot_getvariables(D, vars, {'R_HAND_x','R_HAND_y','R_HAND_z'});
            [traj_xd, traj_yd, traj_zd]     = clmcplot_getvariables(D, vars, {'R_HAND_xd','R_HAND_yd','R_HAND_zd'});
            [traj_xdd, traj_ydd, traj_zdd]  = clmcplot_getvariables(D, vars, {'R_HAND_xdd','R_HAND_ydd','R_HAND_zdd'});
            [traj_q0, traj_q1, traj_q2, traj_q3]	= clmcplot_getvariables(D, vars, {'R_HAND_q0','R_HAND_q1','R_HAND_q2','R_HAND_q3'});
        %     [traj_q0d, traj_q1d, traj_q2d, traj_q3d]    = clmcplot_getvariables(D, vars, {'R_HAND_q0d','R_HAND_q1d','R_HAND_q2d','R_HAND_q3d'});
        %     [traj_q0dd, traj_q1dd, traj_q2dd, traj_q3dd]= clmcplot_getvariables(D, vars, {'R_HAND_q0dd','R_HAND_q1dd','R_HAND_q2dd','R_HAND_q3dd'});
            [traj_ad,  traj_bd,  traj_gd]   = clmcplot_getvariables(D, vars, {'R_HAND_ad','R_HAND_bd','R_HAND_gd'});
            [traj_add, traj_bdd, traj_gdd]  = clmcplot_getvariables(D, vars, {'R_HAND_add','R_HAND_bdd','R_HAND_gdd'});
        %     cart_coord_traj         = [time, traj_x, traj_y, traj_z, traj_xd, traj_yd, traj_zd, traj_xdd, traj_ydd, traj_zdd];
        %     cart_quat_traj          = [time, traj_q0, traj_q1, traj_q2, traj_q3, traj_q0d, traj_q1d, traj_q2d, traj_q3d, traj_q0dd, traj_q1dd, traj_q2dd, traj_q3dd];
        %     cart_quat_ABGomega_traj = [time, traj_q0, traj_q1, traj_q2, traj_q3, traj_ad,  traj_bd,  traj_gd, traj_add, traj_bdd, traj_gdd];
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
        %     [traj_scraping_board_xyz]   = clmcplot_getvariables(D, vars, {'scraping_board_x','scraping_board_y','scraping_board_z'});
        %     [traj_scraping_board_qwxyz] = clmcplot_getvariables(D, vars, {'scraping_board_qw','scraping_board_qx','scraping_board_qy','scraping_board_qz'});
        %     [traj_tool_adaptor_xyz]     = clmcplot_getvariables(D, vars, {'tool_adaptor_x','tool_adaptor_y','tool_adaptor_z'});
        %     [traj_tool_adaptor_qwxyz]   = clmcplot_getvariables(D, vars, {'tool_adaptor_qw','tool_adaptor_qx','tool_adaptor_qy','tool_adaptor_qz'});
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
            [traj_ct_ccdmp_x, traj_ct_ccdmp_y, traj_ct_ccdmp_z] = clmcplot_getvariables(D, vars, {'ct_ccdmp_x','ct_ccdmp_y','ct_ccdmp_z'});
            [traj_ct_qdmp_a, traj_ct_qdmp_b, traj_ct_qdmp_g]    = clmcplot_getvariables(D, vars, {'ct_qdmp_a','ct_qdmp_b','ct_qdmp_g'});
            [traj_supposed_ct_ccdmp_x, traj_supposed_ct_ccdmp_y, traj_supposed_ct_ccdmp_z]  = clmcplot_getvariables(D, vars, {'supposed_ct_ccdmp_x','supposed_ct_ccdmp_y','supposed_ct_ccdmp_z'});
            [traj_supposed_ct_qdmp_a, traj_supposed_ct_qdmp_b, traj_supposed_ct_qdmp_g]     = clmcplot_getvariables(D, vars, {'supposed_ct_qdmp_a','supposed_ct_qdmp_b','supposed_ct_qdmp_g'});
            traj_X_vector   = clmcplot_getvariables(D, vars, getDataNamesWithNumericIndex('X_','vector_',[0:44]));

            %% Clipping Points Determination

            traj_supposed_ct_is_non_zero    = (traj_supposed_ct_ccdmp_x ~= 0) | (traj_supposed_ct_ccdmp_y ~= 0) | (traj_supposed_ct_ccdmp_z ~= 0) |...
                                              (traj_supposed_ct_qdmp_a ~= 0)  | (traj_supposed_ct_qdmp_b ~= 0)  | (traj_supposed_ct_qdmp_g ~= 0);

            [ start_clipping_idx, end_clipping_idx ] = getNullClippingIndex( traj_supposed_ct_is_non_zero );
            % coupling term starts with a zero, so:
            start_clipping_idx      = start_clipping_idx - 1;
            if (start_clipping_idx < 1)
                keyboard;
            end

            initial_final_null_clipped_traj_ct_is_non_zero  = traj_supposed_ct_is_non_zero(start_clipping_idx:end_clipping_idx,1);
            relative_prims_start_indices= find(initial_final_null_clipped_traj_ct_is_non_zero == 0);
            if (length(relative_prims_start_indices) ~= N_prims)    % something is wrong here
                keyboard;
            end
            relative_prims_end_indices  = zeros(size(relative_prims_start_indices));
            relative_prims_end_indices(1:N_prims-1,1)   = relative_prims_start_indices(2:N_prims,1) - 1;
            relative_prims_end_indices(N_prims,1)       = size(initial_final_null_clipped_traj_ct_is_non_zero,1);

            prims_start_indices = start_clipping_idx - 1 + relative_prims_start_indices;
            prims_end_indices   = start_clipping_idx - 1 + relative_prims_end_indices;

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
                       , zeros(length(time), 4) ...     % traj_tool_adaptor_qwxyz ...            % idx   90-93
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
                       , traj_ct_ccdmp_x ...                    % idx     154 (Coupling Term, Position, x)
                       , traj_ct_ccdmp_y ...                    % idx     155 (Coupling Term, Position, y)
                       , traj_ct_ccdmp_z ...                    % idx     156 (Coupling Term, Position, z)
                       , traj_ct_qdmp_a ...                     % idx     157 (Coupling Term, Orientation, alpha)
                       , traj_ct_qdmp_b ...                     % idx     158 (Coupling Term, Orientation, beta)
                       , traj_ct_qdmp_g ...                     % idx     159 (Coupling Term, Orientation, gamma)
                       , traj_supposed_ct_ccdmp_x ...           % idx     160 (Supposed Coupling Term, Position, x)
                       , traj_supposed_ct_ccdmp_y ...           % idx     161 (Supposed Coupling Term, Position, y)
                       , traj_supposed_ct_ccdmp_z ...           % idx     162 (Supposed Coupling Term, Position, z)
                       , traj_supposed_ct_qdmp_a ...            % idx     163 (Supposed Coupling Term, Orientation, alpha)
                       , traj_supposed_ct_qdmp_b ...            % idx     164 (Supposed Coupling Term, Orientation, beta)
                       , traj_supposed_ct_qdmp_g ...            % idx     165 (Supposed Coupling Term, Orientation, gamma)
                       , traj_X_vector ...                      % idx 166-210 (X_vector, 45-dimensional)
                       ];

            for np = 1:N_prims
                out_prim_dir_path   = [out_data_dir_path,'/prim',...
                                       num2str(np,'%02d'),'/'];
                createDirIfNotExist(out_prim_dir_path);

                sm_primitive_traj 	= sm_traj(prims_start_indices(np,1):prims_end_indices(np,1),:);
                if (size(sm_primitive_traj,1) ~= size(sample_traj_cell{np, 1},1))   % primitive size must be (by construction) equal in size with the sample primitive!!!
                    keyboard;
                end
                if (all(sm_primitive_traj(1, 154:165) == 0) == 0)   % ALL initial coupling term for each primitive is supposed to be ZERO!!!
                    keyboard;
                end
                out_data_file_path  = [out_prim_dir_path,'/',...
                                       num2str(data_file_count,'%02d'),'.txt'];
                dlmwrite(out_data_file_path, sm_primitive_traj, ...
                         'delimiter', ' ');
            end
            
            data_file_count         = data_file_count + 1;
        end
    end
end