% Author: Giovanni Sutanto
% Date  : Aug 2019

close   all;
clear  	all;
clc;

generic_task_type   = 'scraping';
specific_task_type 	= 'scraping_w_tool';

[~, raw_hostname]   = system('hostname');
hostname            = raw_hostname(1:end-1);

root_dmp_suffix_path= 'workspace/src/catkin/planning/amd_clmc_dmp/';
if (strcmp(hostname, 'amdgsutanto-XPS-15-9560') == 1)
    home_path       = '/home/amdgsutanto/';
    root_dmp_path   = [home_path, 'AMD_CLMC/Repo/amd_clmc_arm/', root_dmp_suffix_path];
elseif (strcmp(hostname, 'arm') == 1)
    username        = getenv('USER');
    home_path       = ['/home/', username, '/'];
    root_dmp_path   = [home_path, 'Software/', root_dmp_suffix_path];
else
	error('Undefined hostname!');
end
matlab_ltacfb_path  = [root_dmp_path, 'matlab/dmp_coupling/learn_tactile_feedback/'];
python_ltacfb_path  = [root_dmp_path, 'python/dmp_coupling/learn_tactile_feedback/'];

addpath([matlab_ltacfb_path, '../../utilities/']);
addpath([matlab_ltacfb_path, '../../utilities/clmcplot/']);
addpath([matlab_ltacfb_path, '../../utilities/quaternion/']);
addpath([matlab_ltacfb_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([matlab_ltacfb_path, '../../cart_dmp/quat_dmp/']);
addpath([matlab_ltacfb_path, '../../dmp_multi_dim/']);
addpath([matlab_ltacfb_path, '../../neural_nets/feedforward/pmnn/']);

date                = '20190730';
addtnl_description 	= '_on_barrett_hand_208_trained_on_settings_4RLto6RL_rl_on_setting_8_reg_hidden_layer_100';

experiment_name     = [date,'_',specific_task_type,'_correctable',addtnl_description];

data_root_dir_path      = ['~/Desktop/dmp_robot_unroll_results/',generic_task_type,'/',experiment_name,'/'];
in_data_root_dir_path   = [data_root_dir_path,'robot/'];
out_data_root_dir_path  = [data_root_dir_path,'processed/'];

if (~exist(in_data_root_dir_path, 'dir'))
    error([in_data_root_dir_path, ' does NOT exist!']);
end

python_learn_tactile_fb_models_dir_path = [python_ltacfb_path, 'models/'];

reinit_selection_idx= dlmread([python_learn_tactile_fb_models_dir_path, 'reinit_selection_idx.txt']);
TF_max_train_iters 	= dlmread([python_learn_tactile_fb_models_dir_path, 'TF_max_train_iters.txt']);

unroll_types            = {'bsln', 'cpld_before_rl', 'cpld_after_rl'};   % 'bsln' = baseline; 'cpld_before_rl' = coupled before RL; 'cpld_after_rl' = coupled after RL

N_prims                 = 3;

%% Dictionary Definition/Mapping

dict            = containers.Map;
dict('p1_25')   = 1;
dict('p2_5')    = 2;
dict('p3_75')   = 3;
dict('p5')      = 4;
dict('p6_25')   = 5;
dict('p7_5')    = 6;
dict('p8_75')   = 7;
dict('p10')     = 8;

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

%% Extracting and Segmenting Data by Primitive
for nut = 1:size(unroll_types, 2)
    out_data_unroll_type_root_dir_path  = [out_data_root_dir_path, unroll_types{1, nut}, '/'];
    recreateDir(out_data_unroll_type_root_dir_path);
    for ns = 1:N_settings_considered
        %% Paths Determination
        exp_desc_subpath    = [in_data_root_dir_path, in_data_dir_names{1,ns}, '/', unroll_types{1, nut}, '/'];
        out_data_dir_path   = [out_data_unroll_type_root_dir_path, out_data_dir_names{1,ns}, '/'];
        recreateDir(out_data_dir_path);

        %% Data Loading
        data_files              = dir([exp_desc_subpath,'/d*']);
        data_file_count         = 1;
        for data_file = data_files'
            in_data_file_path   = [exp_desc_subpath,'/',data_file.name];
            fprintf('Processing %s...\n', in_data_file_path);
            [D,vars,freq]       = clmcplot_convert(in_data_file_path);
            dt                  = 1.0/freq;

            time    = clmcplot_getvariables(D, vars, {'time'});
            prim_id = clmcplot_getvariables(D, vars, {'ul_curr_prim_no'});
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
            
            prims_start_indices = zeros(N_prims, 1);
            prims_end_indices   = zeros(N_prims, 1);
            for np = 1:N_prims
                prim_indices    = find(prim_id == np - 1);
                if (any((prim_indices(2:end) - prim_indices(1:end-1)) ~= 1))    % something is wrong here, because supposedly there is no jumping indices
                    keyboard;
                end
                prims_start_indices(np,1)   = prim_indices(1) + 1;
                prims_end_indices(np,1)     = prim_indices(end) + 1;
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

is_plotting_other_sensings         	= 0;
other_sense_strings                 = {'global force'};
dict                                = containers.Map;
dict('position')                    = 2;
dict('local force')                 = 4;
dict('local torque')                = 5;
dict('global force')                = 19;
dict('global torque')               = 20;
dict('FT control PI gating gains')  = 21;

is_using_joint_sensing      = 1;        % or proprioceptive sensing
BT_electrode_data_idx       = [6, 14];

is_using_R_LF_electrodes    = 1;

if (strcmp(specific_task_type, 'scraping_w_tool') == 1)
	is_using_R_RF_electrodes  				= 1;
elseif (strcmp(specific_task_type, 'scraping_wo_tool') == 1)
	is_using_R_RF_electrodes  				= 0;
else
	error('Unknown specific_task_type specification!');
end

N_plots                     = 0;
X_plot_group                = cell(0);
plot_titles                 = cell(0);
save_titles                 = cell(0);
if (is_using_R_LF_electrodes)
    N_plots                 = N_plots + 1;
    X_plot_group{1,N_plots} = [1:19];
    plot_titles{1,N_plots}  = 'R\_LF\_electrode';
    save_titles{1,N_plots}  = 'R_LF_electrode';
end
if (is_using_R_RF_electrodes)
    N_plots                 = N_plots + 1;
    X_plot_group{1,N_plots} = [20:38];
    plot_titles{1,N_plots}  = 'R\_RF\_electrode';
    save_titles{1,N_plots}  = 'R_RF_electrode';
end

is_plotting_select_modalities_for_publication   = 1;
select_prim                                     = 2;
select_X_plot_group                             = {[1],[25]};
select_plot_titles                              = {'Left BioTac Finger electrode', 'Right BioTac Finger electrode'};
select_robot_unroll_trial_no                    = 3;
select_font_size                                = 50;
select_line_width                               = 5;

%% Data Loading

load([matlab_ltacfb_path, 'dmp_baseline_params_',generic_task_type,'.mat']);
load([matlab_ltacfb_path, 'data_demo_',generic_task_type,'.mat']);
load([matlab_ltacfb_path, 'dataset_Ct_tactile_asm_',generic_task_type,'_augmented.mat']);

N_settings   	= size(data_demo.coupled, 2);
N_prims         = size(data_demo.coupled, 1);
N_fingers       = size(BT_electrode_data_idx, 2);
N_electrodes    = size(data_demo.coupled{1,1}{1,6},2);
if (is_using_joint_sensing)
    N_joints    = size(data_demo.coupled{1,1}{1,22},2);
end

[ data_temp ]   = extractSensoriMotorTracesSettingsRobotUnroll( [out_data_root_dir_path, 'bsln/'], N_settings + 1, N_prims );
data_robot_unroll.bsln              = data_temp.coupled;

[ data_temp ]	= extractSensoriMotorTracesSettingsRobotUnroll( [out_data_root_dir_path, 'cpld_before_rl/'], N_settings + 1, N_prims );
data_robot_unroll.cpld_before_rl    = data_temp.coupled;

[ data_temp ]	= extractSensoriMotorTracesSettingsRobotUnroll( [out_data_root_dir_path, 'cpld_after_rl/'], N_settings + 1, N_prims );
data_robot_unroll.cpld_after_rl     = data_temp.coupled;

save([out_data_root_dir_path, 'data_robot_unroll_',generic_task_type,'.mat'],'data_robot_unroll');

% end of Data Loading

%% Enumeration of Settings Tested/Unrolled by the Robot

robot_unrolled_settings     = cell(0);

N_unrolled_settings         = 0;
for setting_number = 1:N_settings
    if (~isempty(data_robot_unroll.cpld_before_rl{1,setting_number}))
        N_unrolled_settings = N_unrolled_settings + 1;
        robot_unrolled_settings{1,N_unrolled_settings} = setting_number;
    end
end

% end of Enumeration of Settings Tested/Unrolled by the Robot

for n_unrolled_settings=1:N_unrolled_settings
    setting_no  = robot_unrolled_settings{1,n_unrolled_settings};

    %% Plotting

%     for np=1:N_prims
    for np=2
        N_demo                  = size(dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}, 1);
        N_robot_unroll_trials 	= size(data_robot_unroll.cpld_before_rl{np,setting_no}, 1);
        traj_length           	= size(data_robot_unroll.cpld_before_rl{np,setting_no}{1,23}, 1);
        
        generalization_test_demo_trial_no   = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(generalization_test_demo_trial_rank_no,1);
        
        D_input             = size(dataset_Ct_tactile_asm.sub_X{np,1}{1,1}, 2);
        regular_NN_hidden_layer_topology = dlmread([python_learn_tactile_fb_models_dir_path, 'regular_NN_hidden_layer_topology.txt']);
        N_phaseLWR_kernels  = size(dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,1}{1,1}, 2);
        D_output            = 6;
        
        regular_NN_hidden_layer_activation_func_list = readStringsToCell([python_learn_tactile_fb_models_dir_path, 'regular_NN_hidden_layer_activation_func_list.txt']);

        NN_info.name        = 'my_ffNNphaseLWR';
        NN_info.topology    = [D_input, regular_NN_hidden_layer_topology, N_phaseLWR_kernels, D_output];
        NN_info.activation_func_list= {'identity', regular_NN_hidden_layer_activation_func_list{:}, 'identity', 'identity'};
        NN_info.filepath    = [model_path, 'prim_', num2str(np), '_params_reinit_', num2str(reinit_selection_idx(1, np)), '_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];

        % Perform PMNN Coupling Term Prediction
        [ Ct_prediction, ~ ]= performPMNNPrediction( NN_info, ...
                                                     dataset_Ct_tactile_asm.sub_X{np,setting_no}{generalization_test_demo_trial_no,1}, ...
                                                     dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,setting_no}{generalization_test_demo_trial_no,1} );

        for nxplot=1:size(X_plot_group, 2)
            figure;
            X_plot_indices  = X_plot_group{1, nxplot};
            D               = length(X_plot_indices);
            for d=1:D
                data_idx    = X_plot_indices(d);
                N_plot_cols = ceil(D/5);
                subplot(ceil(D/N_plot_cols),N_plot_cols,d);
                hold on;
                    for nd=1:size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1)
                        demo_idx                    = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                        X_dim_demo_traj         	= dataset_Ct_tactile_asm.sub_X{np,setting_no}{demo_idx,1}(:,data_idx);
                        stretched_X_dim_demo_traj   = stretchTrajectory( X_dim_demo_traj', traj_length )';
                        if (demo_idx == generalization_test_demo_trial_no)
                            p_gen_test_demo_X       = plot(stretched_X_dim_demo_traj, 'c','LineWidth',3);
                        else
                            p_training_demos_X 	    = plot(stretched_X_dim_demo_traj, 'b');
                        end
                    end
                    
                    for robot_unroll_trial_no=1:N_robot_unroll_trials
                        if (robot_unroll_trial_no <= 3)
                            p_robot_unroll_baseline_robot_X = plot(data_robot_unroll.bsln{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'g','LineWidth',1);
                        end
                        p_robot_unroll_coupled_robot_X  = plot(data_robot_unroll.cpld_before_rl{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'r','LineWidth',1);
                    end
                    
                    if (d==1)
                        legend([p_training_demos_X, p_gen_test_demo_X, ...
                                p_robot_unroll_baseline_robot_X, p_robot_unroll_coupled_robot_X], ...
                               'training demos sensor trace', 'generalization test demo sensor trace', ...
                               'robot unroll without ct (baseline) sensor trace', 'robot unroll with ct robot-computed sensor trace');
                    end
                    xlabel(['data\_idx=', num2str(d)]);
                hold off;
            end
            set(gcf,'NextPlot','add');
            axes;
            h   = title([plot_titles{1,nxplot},', prim #',num2str(np), ', setting #', num2str(setting_no)]);
            set(gca,'Visible','off');
            set(h,'Visible','on');
        end
        
        % Sensor Traces Deviation Plot for Publication/Paper
        if ((is_plotting_select_modalities_for_publication) && (np == select_prim))
            for nxplot=1:size(select_X_plot_group, 2)
                select_X_plot_indices   = select_X_plot_group{1, nxplot};
                D                       = length(select_X_plot_indices);
                for d=1:D
                    data_idx            = select_X_plot_indices(d);
                    figure('units','normalized','outerposition',[0 0 1 1]);
                    hold on;
                        N_demo_plots    = size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1);
                        DS_demos        = zeros(traj_length, N_demo_plots);
                        for nd=1:N_demo_plots
                            demo_idx                    = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                            X_dim_demo_traj         	= dataset_Ct_tactile_asm.sub_X{np,setting_no}{demo_idx,1}(:,data_idx);
                            stretched_X_dim_demo_traj   = stretchTrajectory( X_dim_demo_traj', traj_length )';
                            p_training_demos_X          = plot(stretched_X_dim_demo_traj, 'b');
                            DS_demos(:,nd)              = stretched_X_dim_demo_traj;
                        end
                        mean_DS                         = mean(DS_demos, 2);
                        std_DS                          = std(DS_demos, 0, 2);
                        upper_std_DS                    = mean_DS + std_DS;
                        lower_std_DS                    = mean_DS - std_DS;
                        
                        plot(mean_DS,'k--','LineWidth',select_line_width);
                        plot(upper_std_DS,'k','LineWidth',select_line_width);
                        plot(lower_std_DS,'k','LineWidth',select_line_width);

                        for robot_unroll_trial_no=select_robot_unroll_trial_no
                            if (robot_unroll_trial_no <= 3)
                                p_robot_unroll_baseline_robot_X = plot(data_robot_unroll.bsln{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'g','LineWidth',select_line_width);
                            end
                            p_robot_unroll_coupled_robot_X  = plot(data_robot_unroll.cpld_before_rl{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'r','LineWidth',select_line_width);
                        end
                        
%                         lgd =   legend([p_training_demos_X, ...
%                                         p_robot_unroll_baseline_robot_X, p_robot_unroll_coupled_robot_X], ...
%                                        'training demos sensor trace', ...
%                                        'robot unroll without ct (baseline) sensor trace', 'robot unroll with ct robot-computed sensor trace');
%                         lgd.FontSize    = 30;
%                         title(['Sensor Trace Deviation or Delta X of ',select_plot_titles{1,nxplot},' #',num2str(data_idx),', prim #',num2str(np), ', setting ', select_angle]);
%                         xlabel('time');
%                         ylabel('Sensor Trace Deviation or Delta X');
                        ylim([-200, 500]);
                        set(gca, 'FontSize', select_font_size);
                    hold off;
                    print(['~/AMD_CLMC/Publications/LearningCtASMPaper/figures/real_robot_experiment/unrolling/sensor_trace/sensor_trace_',...
                           save_titles{1,nxplot},'_',num2str(data_idx),'_prim_',num2str(np),'_setting_',num2str(setting_no)],'-dpng','-r0');
                end
            end
        end

        D                   = 3;
        if (np == 1)
            figure;
            act_type_string     = 'position';
            for d=1:D
                subplot(D,1,d);
                hold on;
                    if (d == 1)
                        dim_string  = 'x';
                    elseif (d == 2)
                        dim_string  = 'y';
                    elseif (d == 3)
                        dim_string  = 'z';
                    end
                    for nd=1:size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1)
                        demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                        demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}{demo_idx,1}(:,d);
                        stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_length).';
                        if (demo_idx == generalization_test_demo_trial_no)
                            p_gen_test_demo_ct_target   = plot(stretched_demo_ct_target_dim, 'c','LineWidth',3);
                        else
                            p_training_demos_ct_target  = plot(stretched_demo_ct_target_dim, 'b');
                        end
                    end
                    demo_ct_unroll_dim                  = Ct_prediction(:,d);
                    stretched_demo_ct_unroll_dim        = stretchTrajectory(demo_ct_unroll_dim.', traj_length).';
                    p_gen_test_demo_ct_unroll           = plot(stretched_demo_ct_unroll_dim, 'm','LineWidth',3);
                    for robot_unroll_trial_no=1:N_robot_unroll_trials
                        if (robot_unroll_trial_no <= 3)
                            p_robot_unroll_unapplied_coupled_ct = plot(data_robot_unroll.bsln{np,setting_no}{robot_unroll_trial_no,24}(:,d), 'g','LineWidth',1);
                        end
                        p_robot_unroll_coupled_ct    	= plot(data_robot_unroll.cpld_before_rl{np,setting_no}{robot_unroll_trial_no,23}(:,d), 'r','LineWidth',1);
                    end
                    
                    if (d == 1)
                        title([act_type_string, ' coupling term, prim #', num2str(np), ', setting #', num2str(setting_no)]);
                        legend([p_training_demos_ct_target, p_gen_test_demo_ct_target, p_gen_test_demo_ct_unroll, ...
                                p_robot_unroll_coupled_ct, p_robot_unroll_unapplied_coupled_ct], ...
                               'training demos target ct', 'generalization test demo target ct', 'generalization test demo unroll ct', ...
                               ['robot unroll ct ', act_type_string, ' ', dim_string, ' unroll ct'], ...
                               ['robot unroll ct (computed but not applied) ', act_type_string, ' ', dim_string, ' unroll ct']);
                    end
                hold off;
            end
        else
            % Coupling Terms
            if ((is_plotting_select_modalities_for_publication) && (np == select_prim))
                figure('units','normalized','outerposition',[0 0 1 1]);
                d   = 2;
                hold on;
                    if (d == 1)
                        dim_string  = 'pitch';
                    elseif (d == 2)
                        dim_string  = 'roll';
                    elseif (d == 3)
                        dim_string  = 'yaw';
                    end
                    N_demo_plots            = size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1);
                    C_demos                 = zeros(traj_length, N_demo_plots);
                    for nd=1:N_demo_plots
                        demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                        demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}{demo_idx,1}(:,D+d);
                        stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_length).';
                        p_training_demos_ct_target      = plot(stretched_demo_ct_target_dim, 'b');
                     	C_demos(:,nd)                   = stretched_demo_ct_target_dim;
                    end
                    mean_C                              = mean(C_demos, 2);
                    std_C                               = std(C_demos, 0, 2);
                    upper_std_C                         = mean_C + std_C;
                    lower_std_C                         = mean_C - std_C;

                    plot(mean_C,'k--','LineWidth',select_line_width);
                    plot(upper_std_C,'k','LineWidth',select_line_width);
                    plot(lower_std_C,'k','LineWidth',select_line_width);
                    
                    for robot_unroll_trial_no=select_robot_unroll_trial_no
                        p_robot_unroll_coupled_ct     	= plot(data_robot_unroll.cpld_before_rl{np,setting_no}{robot_unroll_trial_no,23}(:,D+d), 'r','LineWidth',select_line_width);
                    end
                    p_robot_unroll_baseline_ct          = plot(zeros(size(data_robot_unroll.cpld_before_rl{np,setting_no}{robot_unroll_trial_no,23}(:,D+d))), 'g','LineWidth',select_line_width);
%                     title([act_type_string, ' coupling term, prim #', num2str(select_prim), ', setting ', select_angle]);
%                     lgd     = legend([p_training_demos_ct_target, p_robot_unroll_coupled_ct], ...
%                                       'training demos target ct', ['robot unroll ct ', act_type_string, ' ', dim_string, ' unroll ct']);
%                     lgd.FontSize    = 30;
%                     xlabel('time');
%                     ylabel('Coupling Term');
                    if (setting_no == 1)
                        lgd     = legend([p_training_demos_ct_target, p_robot_unroll_coupled_ct, p_robot_unroll_baseline_ct], ...
                                          'demos', 'robot unroll with learned coupling term', 'robot unroll without coupling term');
                        lgd.FontSize    = select_font_size;
                    end
                    ylim([-20, 100]);
                    set(gca, 'FontSize', select_font_size);
                hold off;
                print(['~/AMD_CLMC/Publications/LearningCtASMPaper/figures/real_robot_experiment/unrolling/coupling_term_demo_vs_robot_unroll_plot',...
                       '_prim_',num2str(np),'_setting_',num2str(setting_no)],'-dpng','-r0');
            else
                figure;
                act_type_string     = 'orientation';
                for d=1:D
                    subplot(D,1,d);
                    hold on;
                        if (d == 1)
                            dim_string  = 'alpha';
                        elseif (d == 2)
                            dim_string  = 'beta';
                        elseif (d == 3)
                            dim_string  = 'gamma';
                        end
                        for nd=1:size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1)
                            demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                            demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}{demo_idx,1}(:,D+d);
                            stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_length).';
                            if (demo_idx == generalization_test_demo_trial_no)
                                p_gen_test_demo_ct_target   = plot(stretched_demo_ct_target_dim, 'c','LineWidth',3);
                            else
                                p_training_demos_ct_target  = plot(stretched_demo_ct_target_dim, 'b');
                            end
                        end
                        demo_ct_unroll_dim                  = Ct_prediction(:,D+d);
                        stretched_demo_ct_unroll_dim        = stretchTrajectory(demo_ct_unroll_dim.', traj_length).';
                        p_gen_test_demo_ct_unroll           = plot(stretched_demo_ct_unroll_dim, 'm','LineWidth',3);
                        for robot_unroll_trial_no=1:N_robot_unroll_trials
                            if (robot_unroll_trial_no <= 3)
                                p_robot_unroll_unapplied_coupled_ct = plot(data_robot_unroll.bsln{np,setting_no}{robot_unroll_trial_no,24}(:,D+d), 'g','LineWidth',1);
                            end
                            p_robot_unroll_coupled_ct     	= plot(data_robot_unroll.cpld_before_rl{np,setting_no}{robot_unroll_trial_no,23}(:,D+d), 'r','LineWidth',1);
                        end
                        if (d == 1)
                            title([act_type_string, ' coupling term, prim #', num2str(np), ', setting #', num2str(setting_no)]);
                            legend([p_training_demos_ct_target, p_gen_test_demo_ct_target, p_gen_test_demo_ct_unroll, ...
                                    p_robot_unroll_coupled_ct, p_robot_unroll_unapplied_coupled_ct], ...
                                   'training demos target ct', 'generalization test demo target ct', 'generalization test demo unroll ct', ...
                                   ['robot unroll ct ', act_type_string, ' ', dim_string, ' unroll ct'], ...
                                   ['robot unroll ct (computed but not applied) ', act_type_string, ' ', dim_string, ' unroll ct']);
                        end
                    hold off;
                end
            end
        end
        
        if (is_plotting_other_sensings)
            for n_os = 1:length(other_sense_strings)
                figure;
                other_sense_string              = other_sense_strings{n_os};
                for d=1:D
                    subplot(D,1,d);
                    hold on;
                        for nd=1:size(data_demo.coupled{np,setting_no}, 1)
                            demo_other_sense_dim            = data_demo.coupled{np,setting_no}{nd,dict(other_sense_string)}(:,d);
                            stretched_demo_other_sense_dim  = stretchTrajectory(demo_other_sense_dim.', traj_length).';
                            p_training_demos_other_sense    = plot(stretched_demo_other_sense_dim, 'b');
                        end
                        for nd=1:size(data_robot_unroll.bsln{np,setting_no}, 1)
                            p_robot_unroll_baseline_other_sense = plot(data_robot_unroll.bsln{np,setting_no}{nd,dict(other_sense_string)}(:,d), 'g');
                        end
                        for nd=1:size(data_robot_unroll.cpld_before_rl{np,setting_no}, 1)
                            p_robot_unroll_coupled_other_sense  = plot(data_robot_unroll.cpld_before_rl{np,setting_no}{nd,dict(other_sense_string)}(:,d), 'r');
                        end
                        if (d == 1)
                            title([other_sense_string, ' sensing profile, prim #', num2str(np), ', setting #', num2str(setting_no)]);
                            legend([p_training_demos_other_sense, ...
                                    p_robot_unroll_baseline_other_sense, p_robot_unroll_coupled_other_sense], ...
                                   'training demos sensor trace', ...
                                   'robot unroll without ct (baseline) sensor trace', 'robot unroll with ct sensor trace');
                        end
                    hold off;
                end
            end
        end
    end

    % end of Plotting
end