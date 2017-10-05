% Author: Giovanni Sutanto
% Date  : July 28, 2017

close all;
clear all;
clc;

addpath('../../../matlab/utilities/');
addpath('../../../matlab/utilities/clmcplot/');
addpath('../../../matlab/utilities/quaternion/');

specific_task_type  = 'scraping_w_tool';

% Change in_data_dir_path to the path of the input data directory, as
% necessary:
in_data_dir_path    = ['~/Desktop/dmp_demos/',specific_task_type,'/human_baseline/'];
out_data_dir_path   = [pwd, '/../../../../amd_clmc_dmp_data/dmp_coupling/learn_tactile_feedback/',specific_task_type,'/human_baseline/'];

% tool_adaptor is the tip of the transformation chain   (is_tool_adaptor_tip_of_transform_chain == 1)
% end-effector is the tip of the transformation chain   (is_tool_adaptor_tip_of_transform_chain == 0)
is_tool_adaptor_tip_of_transform_chain  = 0;

is_plotting_zero_crossing               = 0;
is_plotting_alignment_matching          = 0;
is_visualizing_traj_alignment           = 0;
is_plotting_excluded_demo_segmentation  = 1;
is_debugging_failed_segmentation        = 0;

% is_baseline_first_to_be_processed   = 1;
is_using_WLS_for_alignment          = 1;    % is using Weighted-Least-Squares for alignment

N_primitive                         = 3;
max_num_coupled_settings_considered = 16;

extra_align_check_length_proportion = 1.25; % (relative to the given/original length)
max_dtw_input_traj_length           = -1;
% max_dtw_input_traj_length           = 770;

% low-pass filter parameters:
fc          = 1.0;

% Dimensions z (column 4) and y (column 3) of end-effector position trajectory 
% (in global coordinate system) will be used 
% for alignment of primitive 1 and primitive 3, respectively.
% This is because signals/trajectories in these axes are the most
% consistent in the context of scraping demonstrations
% (signal in x-axis is not consistent...):
dim_alignment_traj_cell  = {[4],[],[3]}.'; % use y and z

is_segmentation_successful  = 0;

recreateDir(out_data_dir_path);
data_files      = dir(strcat(in_data_dir_path,'/','d*'));
data_file_count = 1;
for data_file = data_files'
    in_data_file_path  = [in_data_dir_path,'/',data_file.name];
    fprintf('Processing %s...\n', in_data_file_path);
    [D,vars,freq]   = clmcplot_convert(in_data_file_path);

    time    = clmcplot_getvariables(D, vars, {'time'});
    [traj_x, traj_y, traj_z]        = clmcplot_getvariables(D, vars, {'R_HAND_x','R_HAND_y','R_HAND_z'});
    [traj_xd, traj_yd, traj_zd]     = clmcplot_getvariables(D, vars, {'R_HAND_xd','R_HAND_yd','R_HAND_zd'});
    [traj_xdd, traj_ydd, traj_zdd]  = clmcplot_getvariables(D, vars, {'R_HAND_xdd','R_HAND_ydd','R_HAND_zdd'});
    [traj_q0, traj_q1, traj_q2, traj_q3]        = clmcplot_getvariables(D, vars, {'R_HAND_q0','R_HAND_q1','R_HAND_q2','R_HAND_q3'});
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

    dt  = 1.0/freq;

    %% Clipping based on End-Effector Velocity in Global z-Axis

    [ zd_based_clip_retain_idx_cell ] = getDataClippingRetainIndex( ...
                                            traj_zd, (is_plotting_zero_crossing && (data_file_count == 1)), ...
                                            0.05, 0.05, 1, dt, 1, 0, fc ); % for scraping_w_tool
%                                             0.015, 0.015, 1, dt, 1, 0, fc ); % for scraping_wo_tool

    %% Clipping based on End-Effector Velocity in Global y-Axis

    % Turn off all global y-axis velocity signals before the end-clipping
    % point of the previous global z-axis velocity-based clipping:
    modified_traj_yd                                            = traj_yd;
    modified_traj_yd(1:zd_based_clip_retain_idx_cell{1,1}(end)) = 0;

    [ yd_based_clip_retain_idx_cell ] = getDataClippingRetainIndex( ...
                                            modified_traj_yd, (is_plotting_zero_crossing && (data_file_count == 1)), ...
                                            0.02, 0.02, 1, dt, 1, 0, fc ); % for scraping_w_tool
%                                             0.0075, 0.0075, 1, dt, 1, 0, fc ); % for scraping_wo_tool

    %% Clipping based on End-Effector Velocity

    traj_endeff_velocity    = [traj_xd, traj_yd, traj_zd];

%   [ velocity_based_clip_retain_idx_cell ] = getDataClippingRetainIndex( ...
%                                                   traj_endeff_velocity, (is_plotting_zero_crossing && (data_file_count == 1)), ...
%                                                 	0.035, 0.035, 1, dt, 1, 0, fc );

    %% Clipping based on End-Effector Omega (Angular Velocity)

    % Turn off all omega signals before the end-clipping
    % point of the previous global z-axis velocity-based clipping,
    % and after the end-clipping point of the previous global y-axis
    % velocity-based clipping:
% 	  traj_endeff_omega           = [traj_ad,  traj_bd,  traj_gd];
%     modified_traj_endeff_omega  = traj_endeff_omega;
%     modified_traj_endeff_omega(1:zd_based_clip_retain_idx_cell{1,1}(end),:)  = 0;
%     modified_traj_endeff_omega(yd_based_clip_retain_idx_cell{1,1}(1):end,:)  = 0;
%             
%     figure;
%     for omega_dim=1:3
%         subplot(3,1,omega_dim);
%         hold on;
%             plot(traj_endeff_omega(:,omega_dim), 'b');
%             plot([zd_based_clip_retain_idx_cell{1,1}(end)+1, yd_based_clip_retain_idx_cell{1,1}(1)-1], 0, 'ro',...
%                  'LineWidth',3, 'MarkerSize', 10,...
%                  'MarkerFaceColor','r');
%         hold off;
%     end
% 
%     [ omega_based_clip_retain_idx_cell ] = getDataClippingRetainIndex( ...
%                                                 modified_traj_endeff_omega, (is_plotting_zero_crossing && (data_file_count == 1)), ...
%                                                 0.015, 0.015, 1, dt, 1, 0, fc );

    %% Clipping based on R_LF_force Sensing

%     traj_R_LF_force     = [  traj_R_LF_FX-mean(traj_R_LF_FX(1:20))...
%                            , traj_R_LF_FY-mean(traj_R_LF_FY(1:20))...
%                            , traj_R_LF_FZ-mean(traj_R_LF_FZ(1:20))];
% 
%     [ retain_idx_R_LF_force_cell ]  = getDataClippingRetainIndex( ...
%                                                 traj_R_LF_force, (is_plotting_zero_crossing && (data_file_count == 1)), ...
%                                                 0.2, 0.035, 1);
%     if (size(retain_idx_R_LF_force_cell, 2) ~= 1)
%         keyboard;
%     end

    %% Clipping based on BioTac Fingers' Electrodes

%     traj_R_electrodes   = [traj_R_LF_electrodes, traj_R_RF_electrodes];
% 
%     [ R_electrodes_based_clip_retain_idx_cell ] = getDataClippingRetainIndex( ...
%                                                     traj_R_electrodes, (is_plotting_zero_crossing && (data_file_count == 1)), ...
%                                                     200.0, 200.0, 1, dt, 0, 0, fc, 1);
% 
%     if (size(R_electrodes_based_clip_retain_idx_cell, 2) ~= 1)
%         keyboard;
%     end

    %% Store the Initial Joint States (especially Posture)
     % for Later Deployment on the Real Robot

    if (data_file_count == 1)
        traj_joints     = clmcplot_getvariables(D, vars, ...
                            getDataNamesFromCombinations('R_', ...
                                {'SFE_', 'SAA_', 'HR_', 'EB_', 'WR_', 'WFE_', 'WAA_', 'FR_', 'RF_', 'MF_', 'LF_'}, ...
                                {'th', 'thd', 'thdd'}));
        joint_start_states  = [time(1,1), traj_joints(1,:)];
        out_joint_start_states_file_path   = [out_data_dir_path,'/joint_start_states.txt'];
        dlmwrite(out_joint_start_states_file_path, joint_start_states, ...
                 'delimiter', ' ');
    end

    %% Stack the Data together into a Huge Matrix

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
               ];

    %% Alignment/Matching and Cutting of the Primitives

    initial_prim_idx_cell           = cell(1, N_primitive);
    initial_prim_idx_cell{1,1}      = zd_based_clip_retain_idx_cell{1,1};
    initial_prim_idx_cell{1,2}      = [zd_based_clip_retain_idx_cell{1,1}(end)+1:yd_based_clip_retain_idx_cell{1,1}(1)-1];
    initial_prim_idx_cell{1,3}      = yd_based_clip_retain_idx_cell{1,1};

    % reference trajectory is the first demonstration/trial in
    % baseline setting:
    if (data_file_count == 1)
        ref_traj_unclipped_cell         = cell(N_primitive, 1);

        ref_traj_cell                   = cell(N_primitive, 1);
        ref_traj_clipping_points_cell   = cell(N_primitive, 1);

        ref_traj_extended_cell                  = cell(N_primitive, 1);
        ref_traj_extended_clipping_points_cell  = cell(N_primitive, 1);

        traj_unclipped_cell             = cell(N_primitive, 1);
        traj_clipping_points_cell       = cell(N_primitive, 1);

        current_traj_cell               = cell(N_primitive, 1);
    end

    primitive_processing_order      = [1,3,2];

    for np_idx = 1:N_primitive
        np                          = primitive_processing_order(np_idx);
        out_prim_dir_path           = [out_data_dir_path,'/prim',...
                                       num2str(np,'%02d'),'/'];
        if (~exist(out_prim_dir_path, 'dir'))
            mkdir(out_prim_dir_path);
        end

        if ((data_file_count == 1) || ...
            (~(data_file_count == 1) && ((np == 1) || (np == 3))))
            extra_align_check_length    = round(extra_align_check_length_proportion * (initial_prim_idx_cell{1,np}(end) - initial_prim_idx_cell{1,np}(1)));
        end

        % 1st trial in the setting is used as reference trajectory
        % for alignment of subsequent trials/demonstrations in the same setting:
        if (data_file_count == 1)
            ref_traj_unclipped_cell{np,1}   = sm_traj(:, dim_alignment_traj_cell{np,1});

            % segmented sensorimotor trajectory primitive:
%             sm_primitive_traj       = sm_traj(retain_idx_R_LF_force_cell{1,np}(1):retain_idx_cell{1,np}(end),:);
%             sm_primitive_traj       = sm_traj(retain_idx_cell{1,np}(1):retain_idx_cell{1,np}(end),:);
%             sm_primitive_traj       = sm_traj(R_electrodes_based_clip_retain_idx_cell{1,np}(1):retain_idx_cell{1,np}(end),:);
            ref_traj_clipping_points_cell{np,1} = [initial_prim_idx_cell{1,np}(1), initial_prim_idx_cell{1,np}(end)];
            ref_traj_extended_start_idx = max(1, initial_prim_idx_cell{1,np}(1) - extra_align_check_length);
            ref_traj_extended_end_idx   = min(initial_prim_idx_cell{1,np}(end) + extra_align_check_length, size(sm_traj,1));
            ref_traj_extended_cell{np,1}= sm_traj(ref_traj_extended_start_idx:ref_traj_extended_end_idx, dim_alignment_traj_cell{np,1});
            ref_traj_extended_clipping_points_cell{np,1}= [ref_traj_extended_start_idx, ref_traj_extended_end_idx];
            sm_primitive_traj       = sm_traj(initial_prim_idx_cell{1,np},:);
            % offset the time so that it start with 0:
            sm_primitive_traj(:,1)  = sm_primitive_traj(:,1) - ...
                                      sm_primitive_traj(1,1);
            ref_traj_cell{np,1}     = sm_primitive_traj(:, dim_alignment_traj_cell{np,1});
%             ref_traj_cell           = sm_primitive_traj(:, dim_alignment_traj_cell);

            is_segmentation_successful  = 1;
        else
            if ((np == 1) || (np == 3))
                % trajectory to be aligned:
%                 traj_to_be_aligned  = sm_traj(R_electrodes_based_clip_retain_idx_cell{1,np}(1):retain_idx_cell{1,np}(end), dim_alignment_traj_cell);
                traj_to_be_aligned_extended_start_idx   = max(1, initial_prim_idx_cell{1,np}(1) - extra_align_check_length);
                traj_to_be_aligned_extended_end_idx     = min(initial_prim_idx_cell{1,np}(end) + extra_align_check_length, size(sm_traj,1));
                traj_to_be_aligned_extended             = sm_traj(traj_to_be_aligned_extended_start_idx:traj_to_be_aligned_extended_end_idx, dim_alignment_traj_cell{np,1});
                tic;
%                 [ traj_s, traj_tau ]= alignTrajectory( ref_traj_cell, traj_to_be_aligned, dt );
%                 [ traj_s, traj_tau ]= alignTrajectory( ref_traj_cell{np,1}, traj_to_be_aligned, dt );
%                 start_clipping_index= traj_to_be_aligned_start_idx - 1 + round(traj_s);
                [ traj_extended_s, traj_extended_tau ]  = alignTrajectory( ref_traj_extended_cell{np,1}, traj_to_be_aligned_extended, dt, is_using_WLS_for_alignment, is_plotting_alignment_matching, max_dtw_input_traj_length );
                toc;
                start_clipping_extended_index= traj_to_be_aligned_extended_start_idx - 1 + round(traj_extended_s);
                ref_traj_extended_length    = size(ref_traj_extended_cell{np,1}, 1);
                traj_extended_length        = round(traj_extended_tau * (ref_traj_extended_length - 1));
                end_clipping_extended_index = start_clipping_extended_index + traj_extended_length;

%                 ref_traj_length     = size(ref_traj_cell, 1);
                ref_traj_length         = size(ref_traj_cell{np,1}, 1);

                start_extension_length  = round(traj_extended_tau * (ref_traj_clipping_points_cell{np,1}(1) - ref_traj_extended_clipping_points_cell{np,1}(1)));
                end_extension_length    = round(traj_extended_tau * (ref_traj_extended_clipping_points_cell{np,1}(end) - ref_traj_clipping_points_cell{np,1}(end)));

                start_clipping_index    = start_clipping_extended_index + start_extension_length;
                end_clipping_index      = end_clipping_extended_index - end_extension_length;

                traj_unclipped_cell{np,1}       = sm_traj(:, dim_alignment_traj_cell{np,1});
                traj_clipping_points_cell{np,1} = [start_clipping_index, end_clipping_index];

                % segmented sensorimotor trajectory primitive:
                sm_primitive_traj       = sm_traj(start_clipping_index:end_clipping_index,:);
                % offset the time so that it start with 0:
                sm_primitive_traj(:,1)  = sm_primitive_traj(:,1) - ...
                                          sm_primitive_traj(1,1);

                current_traj_cell{np,1} = sm_primitive_traj(:, dim_alignment_traj_cell{np,1});

                if (is_visualizing_traj_alignment)
%                     if (is_visualizing_traj_alignment && (np == 1))
                    N_dim_alignment_traj= length(dim_alignment_traj_cell{np,1});
                    figure;
                    for d=1:N_dim_alignment_traj
                        subplot(N_dim_alignment_traj, 1, d);
                        hold on;
                            title(['Alignment human_baseline/',data_file.name,' Dimension #', num2str(dim_alignment_traj_cell{np,1}(d))]);
                            plot(ref_traj_cell{np,1}(:,d), 'r');
%                             plot(ref_traj_cell(:,d), 'r');
                            stretched_aligned_traj  = stretchTrajectory( sm_primitive_traj(:,dim_alignment_traj_cell{np,1}(d)).', ref_traj_length ).';
                            plot(stretched_aligned_traj, 'b');
                            legend('reference', 'aligned');
                        hold off;
                    end

                    figure;
                    for d=1:N_dim_alignment_traj
                        subplot(N_dim_alignment_traj, 1, d);
                        hold on;
                            title(['Unclipped Trajectory human_baseline/',data_file.name,' Dimension #', num2str(dim_alignment_traj_cell{np,1}(d))]);
                            plot(ref_traj_unclipped_cell{np,1}(:,d), 'r');
                            plot(traj_unclipped_cell{np,1}(:,d), 'b');
                            legend('reference', 'target');
                            plot(ref_traj_clipping_points_cell{np,1}, 0, 'ro',...
                                 'LineWidth',3, 'MarkerSize', 10,...
                                 'MarkerFaceColor','r');
                            plot(traj_clipping_points_cell{np,1}, 0, 'bo',...
                                 'LineWidth',3, 'MarkerSize', 10,...
                                 'MarkerFaceColor','b');
                        hold off;
                    end
                    keyboard;
                end
            elseif (np == 2)
                start_clipping_index    = traj_clipping_points_cell{np-1,1}(end) + 1;
                end_clipping_index      = traj_clipping_points_cell{np+1,1}(1) - 1;
                if (end_clipping_index > start_clipping_index)
                    sm_primitive_traj       = sm_traj(start_clipping_index:end_clipping_index,:);
                    % offset the time so that it start with 0:
                    sm_primitive_traj(:,1)  = sm_primitive_traj(:,1) - ...
                                              sm_primitive_traj(1,1);
                    is_segmentation_successful  = 1;
                else
                    warning(['primitive #',num2str(np),' does NOT exist!!!']);
                    is_segmentation_successful  = 0;

                    % some plotting for debugging by visualization:
                    for np_eval = [1,3]
                        N_dim_alignment_traj= length(dim_alignment_traj_cell{np_eval,1});
                        if (is_plotting_excluded_demo_segmentation)
                            h_exclude   = figure;
                            for d=1:N_dim_alignment_traj
                                subplot(N_dim_alignment_traj, 1, d);
                                hold on;
                                    title(['Alignment human_baseline/',data_file.name,' Dimension #', num2str(dim_alignment_traj_cell{np_eval,1}(d)), ', Primitive #', num2str(np_eval)]);
                                    plot(ref_traj_cell{np_eval,1}(:,d), 'r');
                                    stretched_aligned_traj  = stretchTrajectory( current_traj_cell{np_eval,1}(:,d).', length(ref_traj_cell{np_eval,1}(:,d)) ).';
                                    plot(stretched_aligned_traj, 'b');
                                    legend('reference', 'aligned');
                                hold off;
                            end
                            figure_exclude_save_path  = [in_data_dir_path,'/exclude/',data_file.name,'_prim',num2str(np_eval,'%02d'),'.jpg'];
                            saveas(h_exclude, figure_exclude_save_path);
                            pause(10)
                            close(h_exclude);
                        end

                        % delete primitive 1 and 3 files of current data file:
                        prim_to_be_deleted_file_path = [out_data_dir_path,'/prim',...
                                                        num2str(np_eval,'%02d'),'/',...
                                                        num2str(data_file_count,'%02d'),'.txt'];
                        delete(prim_to_be_deleted_file_path);
                    end

                    if (is_debugging_failed_segmentation)
                        keyboard;
                    end

                    break;
                end
            else
                error('Should never reach here!!!');
            end
        end

        out_data_file_path = [out_prim_dir_path,'/',...
                              num2str(data_file_count,'%02d'),'.txt'];
        dlmwrite(out_data_file_path, sm_primitive_traj, ...
                 'delimiter', ' ');
    end

%     if ((is_plotting_zero_crossing || is_visualizing_traj_alignment) && (data_file_count == 1))
    if ((is_plotting_zero_crossing || is_visualizing_traj_alignment) && (data_file_count == 3))
        break;
    else
        if (is_segmentation_successful)
            data_file_count = data_file_count + 1;
        end
    end
end