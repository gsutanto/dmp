% Author: Giovanni Sutanto
% Date  : February 2017
% Description:
%   Convert segmented demonstrations into supervised tactile feedback
%   dataset.
%   Version 1: Also learn the baseline/nominal action primitives.

clear  	all;
close   all;
clc;

addpath('../../utilities/');
addpath('../../cart_dmp/cart_coord_dmp/');
addpath('../../cart_dmp/quat_dmp/');
addpath('../../dmp_multi_dim/');

% task_type           = 'peg_in_hole_big_cone';
task_type           = 'scraping';
in_data_dir_path    = ['../../../data/dmp_coupling/learn_tactile_feedback/'...
                       ,task_type,'/'];
is_visualize_baseline_data_fitting  = 1;

is_using_joint_sensing  = 1;        % or proprioceptive sensing
BT_electrode_data_idx   = [6, 14];

N_points_ave_init_offset            = 5;
                   
%% Data Loading

[ data_demo ]   = extractSensoriMotorTracesAllSettings( in_data_dir_path );
save(['data_demo_',task_type,'.mat'],'data_demo');

N_settings   	= size(data_demo.coupled, 2);
N_primitive     = size(data_demo.baseline, 1);
N_finger        = size(BT_electrode_data_idx, 2);
N_electrode     = size(data_demo.baseline{1,1}{1,6},2);
if (is_using_joint_sensing)
    N_joints    = size(data_demo.baseline{1,1}{1,22},2);
end

% end of Data Loading

%% BioTac Signal Offset Removal (using Primitive 1's Offset for ALL Primitives)

N_demo  = size(data_demo.baseline{1,1}, 1);
for ndm=1:N_demo
    for nf=1:N_finger
        for np=1:N_primitive
            if (np == 1)    % (using Primitive 1's Initial Averaged Offset for ALL Primitives)
                BT_electrode_signal_offset  = (1.0/N_points_ave_init_offset) * sum(data_demo.baseline{np,1}{ndm,BT_electrode_data_idx(1,nf)}(1:N_points_ave_init_offset,:),1);
            end
            data_demo.baseline{np,1}{ndm,BT_electrode_data_idx(1,nf)}   = data_demo.baseline{np,1}{ndm,BT_electrode_data_idx(1,nf)} - repmat(BT_electrode_signal_offset, size(data_demo.baseline{np,1}{ndm,BT_electrode_data_idx(1,nf)},1), 1);
        end
    end
end

for ns=1:N_settings
    N_demo  = size(data_demo.coupled{1,ns}, 1);
    for ndm=1:N_demo
        for nf=1:N_finger
            for np=1:N_primitive
                if (np == 1)    % (using Primitive 1's Initial Averaged Offset for ALL Primitives)
                    BT_electrode_signal_offset  = (1.0/N_points_ave_init_offset) * sum(data_demo.coupled{np,ns}{ndm,BT_electrode_data_idx(1,nf)}(1:N_points_ave_init_offset,:),1);
                end
                data_demo.coupled{np,ns}{ndm,BT_electrode_data_idx(1,nf)}   = data_demo.coupled{np,ns}{ndm,BT_electrode_data_idx(1,nf)} - repmat(BT_electrode_signal_offset, size(data_demo.coupled{np,ns}{ndm,BT_electrode_data_idx(1,nf)},1), 1);
            end
        end
    end
end

% end of BioTac Signal Offset Removal

%% Fitting Nominal DMPs and Unrolling

dmp_baseline_params.cart_coord  = cell(N_primitive, 1);
dmp_baseline_params.Quat        = cell(N_primitive, 1);
dmp_baseline_params.BT_electrode= cell(N_primitive, N_finger);
if (is_using_joint_sensing)
    dmp_baseline_params.joint_sense = cell(N_primitive, 1);
end
unrolling_result                = cell(N_primitive, 1);

smoothed_data_demo.baseline     = cell(N_primitive, 1);

ctraj_local_coordinate_frame_selection  = 2;    % schaal's ctraj_local_coordinate_frame_selection (see definition in computeCTrajCoordTransforms.m)

n_rfs               = 25;
c_order             = 1;    % using 2nd order canonical system (for Cartesian position and orientation)

n_rfs_sensor        = 25;
c_order_sensor      = 1;    % using 2nd order canonical system

traj_stretch_length = 1500;

% for position and orientation (6D pose), as well as joint trajectories:
percentage_padding_pose_and_joint           = 1.5;
percentage_smoothing_points_pose_and_joint  = 3.0;

% for BioTac electrodes:
percentage_padding_BT_electrode             = 0.5;
percentage_smoothing_points_BT_electrode    = 1.0;

% low-pass filter cutoff frequency:
low_pass_cutoff_freq_pose           =  1.5;
low_pass_cutoff_freq_BT_electrode   = 50.0;
low_pass_cutoff_freq_joint          = 30.0;

for np=1:N_primitive
    if (np == 1)
        smoothing_mode  = 1;    % smooth start only
    elseif (np == N_primitive)
        smoothing_mode  = 2;    % smooth end only
    else
        smoothing_mode  = 0;    % do not smooth
    end
    
    unrolling_result{np,1}      = cell(1,size(data_demo.baseline{np,1},2));
    N_traj                      = size(data_demo.baseline{np,1},1);
    
    %% Cartesian Coordinate DMP Fitting and Unrolling
    
    % Load Cartesian Coordinate Trajectory Demonstrations into Structure:
    cart_coord_traj_demo_set    = cell(3, N_traj);
    dts                         = zeros(1, N_traj);
    smoothed_data_demo.baseline{np,1}   = cell(size(data_demo.baseline{np,1}));
    for nt=1:N_traj
        time        = data_demo.baseline{np,1}{nt,1}';
        XT          = data_demo.baseline{np,1}{nt,2}(:,1:3);
        XdT         = data_demo.baseline{np,1}{nt,2}(:,4:6);
        XddT        = data_demo.baseline{np,1}{nt,2}(:,7:9);

        tau         = time(1,end) - time(1,1);
        traj_length = size(XT,1);
        dt          = tau/(traj_length-1);
    
        cart_coord_traj_demo_set{1, nt} = XT;
        cart_coord_traj_demo_set{2, nt} = XdT;
        cart_coord_traj_demo_set{3, nt} = XddT;
        
        % perform some smoothing:
        [ smoothed_cart_coord_traj ]    = smoothStartEndNDTrajectoryBasedOnPosition( cart_coord_traj_demo_set(:,nt), ...
                                                                                     percentage_padding_pose_and_joint, ...
                                                                                     percentage_smoothing_points_pose_and_joint, ...
                                                                                     smoothing_mode, dt, ...
                                                                                     low_pass_cutoff_freq_pose );
        cart_coord_traj_demo_set(:,nt)  = smoothed_cart_coord_traj;
        smoothed_data_demo.baseline{np,1}{nt,2} = cell2mat(cart_coord_traj_demo_set(:,nt).');
        
        dts(1,nt)   = dt;
    end
    clearvars       time tau traj_length dt XT XdT XddT;
    
    assert(var(dts) < 1e-10, 'Sampling Time (dt) is inconsistent across demonstrated trajectories.');
    mean_dt         = mean(dts);
    cart_coord_dmp_unroll_dt            = mean_dt;
    cart_coord_dmp_is_using_scaling     = [1, 0, 0];
    
    [ cart_coord_dmp_baseline_params, ...
      cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(cart_coord_traj_demo_set, ...
                                                                                         mean_dt, n_rfs, c_order, ...
                                                                                         ctraj_local_coordinate_frame_selection, ...
                                                                                         -1, cart_coord_dmp_unroll_dt, ...
                                                                                         cart_coord_dmp_is_using_scaling);
  
    dmp_baseline_params.cart_coord{np,1}    = cart_coord_dmp_baseline_params;
    unrolling_result{np,1}{1,2} = [cart_coord_dmp_baseline_unroll_global_traj{1,1},...
                                   cart_coord_dmp_baseline_unroll_global_traj{2,1},...
                                   cart_coord_dmp_baseline_unroll_global_traj{3,1}];
    
    % end of Cartesian Coordinate DMP Fitting and Unrolling
    
    %% Quaternion DMP Fitting and Unrolling
    
    % Load Quaternion Trajectory Demonstrations into Structure:
    Quat_traj_demo_set  = cell(3, N_traj);
    for nt=1:N_traj
        QT          = data_demo.baseline{np,1}{nt,3}(:,1:4)';
        omegaT      = data_demo.baseline{np,1}{nt,3}(:,5:7)';
        omegadT     = data_demo.baseline{np,1}{nt,3}(:,8:10)';

        Quat_traj_demo_set{1, nt} = QT;
        Quat_traj_demo_set{2, nt} = omegaT;
        Quat_traj_demo_set{3, nt} = omegadT;
        
        % perform some smoothing:
        [ smoothed_Quat_traj ] = smoothStartEndQuatTrajectoryBasedOnQuaternion( Quat_traj_demo_set(:,nt), ...
                                                                                percentage_padding_pose_and_joint, ...
                                                                                percentage_smoothing_points_pose_and_joint, ...
                                                                                smoothing_mode, mean_dt, ...
                                                                                low_pass_cutoff_freq_pose );
        Quat_traj_demo_set(:,nt)    = smoothed_Quat_traj;
        smoothed_data_demo.baseline{np,1}{nt,3} = cell2mat(Quat_traj_demo_set(:,nt)).';
    end
    clearvars       QT omegaT omegadT;
    
    [ Quat_dmp_baseline_params, ...
      Quat_dmp_baseline_unroll_traj ]   = learnQuatPrimitiveMulti(Quat_traj_demo_set, ...
                                                                  mean_dt, ...
                                                                  n_rfs, ...
                                                                  c_order, ...
                                                                  cart_coord_dmp_baseline_params.mean_tau);
  
    dmp_baseline_params.Quat{np,1}  = Quat_dmp_baseline_params;
    unrolling_result{np,1}{1,3} = [Quat_dmp_baseline_unroll_traj{1,1},...
                                   Quat_dmp_baseline_unroll_traj{4,1},...
                                   Quat_dmp_baseline_unroll_traj{5,1}];
    
    % end of Quaternion DMP Fitting and Unrolling
    
    %% BioTac electrodes DMP Fitting and Unrolling
    
    for nf=1:N_finger
        BT_electrode_traj_demo_set  = cell(1, N_traj);
        for nt=1:N_traj
            BT_electrode_traj_demo_set{1, nt}   = data_demo.baseline{np,1}{nt,BT_electrode_data_idx(1,nf)}';
            
            % perform some smoothing:
            [ smoothed_BT_electrode_prof ] = smoothStartEndNDSensors( BT_electrode_traj_demo_set{1, nt}.', ...
                                                                      percentage_padding_BT_electrode, ...
                                                                      percentage_smoothing_points_BT_electrode, ...
                                                                      smoothing_mode, mean_dt, ...
                                                                      low_pass_cutoff_freq_BT_electrode );
            BT_electrode_traj_demo_set{1, nt}   = smoothed_BT_electrode_prof.';
            smoothed_data_demo.baseline{np,1}{nt,BT_electrode_data_idx(1,nf)}   = BT_electrode_traj_demo_set{1, nt}.';
        end

        [ BT_electrode_dmp_baseline_params, ...
          BT_electrode_dmp_baseline_unroll_traj ]   = learnMultiDimensionalPrimitiveMulti(BT_electrode_traj_demo_set, ...
                                                                                          mean_dt, ...
                                                                                          n_rfs_sensor, ...
                                                                                          c_order_sensor, ...
                                                                                          -1, 0);

        dmp_baseline_params.BT_electrode{np,nf} = BT_electrode_dmp_baseline_params;
        unrolling_result{np,1}{1,BT_electrode_data_idx(1,nf)}   = [BT_electrode_dmp_baseline_unroll_traj{1,1},...
                                                                   BT_electrode_dmp_baseline_unroll_traj{2,1},...
                                                                   BT_electrode_dmp_baseline_unroll_traj{3,1}];
    end
    
    % end of BioTac electrodes DMP Fitting and Unrolling
    
    %% Joint sensing (proprioceptive) DMP Fitting and Unrolling
    
    if (is_using_joint_sensing)
        joint_sense_traj_demo_set   = cell(1, N_traj);
        for nt=1:N_traj
            joint_sense_traj_demo_set{1, nt}    = data_demo.baseline{np,1}{nt,22}';
            
            % perform some smoothing:
            [ smoothed_joint_sense_prof ]   = smoothStartEndNDSensors( joint_sense_traj_demo_set{1, nt}.', ...
                                                                       percentage_padding_pose_and_joint, ...
                                                                       percentage_smoothing_points_pose_and_joint, ...
                                                                       smoothing_mode, mean_dt, ...
                                                                       low_pass_cutoff_freq_joint );
            joint_sense_traj_demo_set{1, nt}    = smoothed_joint_sense_prof.';
            smoothed_data_demo.baseline{np,1}{nt,22}    = joint_sense_traj_demo_set{1, nt}.';
        end

        [ joint_sense_dmp_baseline_params, ...
          joint_sense_dmp_baseline_unroll_traj ]= learnMultiDimensionalPrimitiveMulti(joint_sense_traj_demo_set, ...
                                                                                      mean_dt, ...
                                                                                      n_rfs_sensor, ...
                                                                                      c_order_sensor);

        dmp_baseline_params.joint_sense{np,1}   = joint_sense_dmp_baseline_params;
        unrolling_result{np,1}{1,22}	= [joint_sense_dmp_baseline_unroll_traj{1,1},...
                                           joint_sense_dmp_baseline_unroll_traj{2,1},...
                                           joint_sense_dmp_baseline_unroll_traj{3,1}];
    end
    
    %% Logging Unrolling Results into Files
    
    unroll_cart_coord_dmp_result_dump_dir_path  = [pwd,'/unroll/',task_type...
                                                   ,'/cart_dmp/cart_coord_dmp/'];
    createDirIfNotExist(unroll_cart_coord_dmp_result_dump_dir_path);
    unroll_cart_coord_dmp_result_dump_file_path = [unroll_cart_coord_dmp_result_dump_dir_path...
                                                   ,num2str(np),'.txt'];
    unroll_cart_coord_dmp_result_dump = [  cart_coord_dmp_baseline_unroll_global_traj{4,1}...
                                         , cart_coord_dmp_baseline_unroll_global_traj{1,1}...
                                         , cart_coord_dmp_baseline_unroll_global_traj{2,1}...
                                         , cart_coord_dmp_baseline_unroll_global_traj{3,1}];
    dlmwrite(unroll_cart_coord_dmp_result_dump_file_path, unroll_cart_coord_dmp_result_dump, ...
             'delimiter', ' ');
         
    unroll_quat_dmp_result_dump_dir_path    = [pwd,'/unroll/',task_type...
                                               ,'/cart_dmp/quat_dmp/'];
    createDirIfNotExist(unroll_quat_dmp_result_dump_dir_path);
    unroll_quat_dmp_result_dump_file_path   = [unroll_quat_dmp_result_dump_dir_path...
                                               ,num2str(np),'.txt'];
    unroll_quat_dmp_result_dump = [  Quat_dmp_baseline_unroll_traj{6,1}...
                                   , Quat_dmp_baseline_unroll_traj{1,1}...
                                   , Quat_dmp_baseline_unroll_traj{2,1}...
                                   , Quat_dmp_baseline_unroll_traj{3,1}...
                                   , Quat_dmp_baseline_unroll_traj{4,1}...
                                   , Quat_dmp_baseline_unroll_traj{5,1}];
    dlmwrite(unroll_quat_dmp_result_dump_file_path, unroll_quat_dmp_result_dump, ...
             'delimiter', ' ');
    clearvars       cart_coord_traj_demo_set cart_coord_dmp_baseline_params cart_coord_dmp_baseline_unroll_global_traj dts;
    clearvars       Quat_traj_demo_set Quat_dmp_baseline_params Quat_dmp_baseline_unroll_traj;
    clearvars       BT_electrode_traj_demo_set BT_electrode_dmp_baseline_params BT_electrode_dmp_baseline_unroll_traj;
    if (is_using_joint_sensing)
        clearvars   joint_sense_traj_demo_set joint_sense_dmp_baseline_params joint_sense_dmp_baseline_unroll_traj;
    end
    clearvars       unroll_cart_coord_dmp_result_dump unroll_quat_dmp_result_dump
    
    % end of Logging Unrolling Results into Files
end
save(['dmp_baseline_params_',task_type,'.mat'],'dmp_baseline_params');

% end of Fitting Nominal DMPs and Unrolling

%% Visualizations of Nominal DMPs Fitting and Unrolling

if (is_visualize_baseline_data_fitting)
    for np=1:N_primitive
        N_traj          = size(data_demo.baseline{np,1},1);

        % The columns of plot_titles and plot_indices below corresponds to 
        % the columns of data_demo.baseline{np,1}.
        % plot_indices represents the range of data plotting 
        % (leftmost column of data_demo.baseline{np,1}{nt,nplotgroups})
        % plot_indices [0] means data_demo.baseline{np,1}{:,nplotgroups} will not be
        % plotted:
        plot_titles     = { {}, {'Cartesian Coordinate X','Cartesian Coordinate Xd','Cartesian Coordinate Xdd'} ...
                           ,{'Cartesian Quaternion Q','Cartesian Quaternion omega','Cartesian Quaternion omegad'} ...
                           ,{}, {} ...%,{'R\_HAND FT Force'}, {'R\_HAND FT Torque'} ...
                           ,{'R\_LF\_electrodes'} ...
                           ,{} ...%,{'R\_LF\_TDC'} ...
                           ,{}, {}, {} ...%,{'R\_LF\_TAC'}, {'R\_LF\_PDC'}, {'R\_LF\_PACs'}...
                           ,{},{} ...
                           ,{} ...%,{'R\_LF computed 3D POC'}...
                           ,{'R\_RF\_electrodes'}...
                           ,{} ...%, {'R\_RF\_TDC'} ...
                           ,{}, {}, {} ...%,{'R\_RF\_TAC'}, {'R\_RF\_PDC'}, {'R\_RF\_PACs'}...
                           ,{},{} ...
                           ,{} ...%,{'R\_RF computed 3D POC'}...
                           };
        plot_indices    = { {},{[1:3],[4:6],[7:9]}...
                           ,{[1:4],[5:7],[8:10]}...
                           ,{}, {} ...%,{[1:3]},{[1:3]}...
                           ,{[1:19]}...
                           ,{} ...%,{[1]}...
                           ,{}, {}, {} ...%,{[1]},{[1]},{[1:5]}...
                           ,{},{} ...
                           ,{} ...%,{[1:3]}...
                           ,{[1:19]}...
                           ,{} ...%,{[1]}...
                           ,{}, {}, {} ...%,{[1]},{[1]},{[1:5]}...
                           ,{},{} ...
                           ,{} ...%,{[1:3]}...
                           };
        if (is_using_joint_sensing)
            plot_titles{1,22}   = {'R\_joint\_sense'};
            plot_indices{1,22}  = {[1:7]};
        end

        for nplotgroups=1:size(plot_indices,2)
            if (~isempty(plot_indices{1,nplotgroups}))
                for nplot=1:size(plot_indices{1,nplotgroups},2)
                    figure;
                    D   = plot_indices{1,nplotgroups}{1,nplot}(1,end)-(plot_indices{1,nplotgroups}{1,nplot}(1,1)-1);
                    for d=1:D
                        data_idx    = (plot_indices{1,nplotgroups}{1,nplot}(1,1)-1) + d;
                        N_plot_cols = ceil(D/5);
                        subplot(ceil(D/N_plot_cols),N_plot_cols,d);
                        if (d==1)
                            title([plot_titles{1,nplotgroups}{1,nplot},', Primitive #',num2str(np)]);
                        end
                        hold on;
                        for nt=1:N_traj
                            demo_traj= data_demo.baseline{np,1}{nt,nplotgroups}(:,data_idx);
                            stretched_demo_traj = stretchTrajectory( demo_traj', traj_stretch_length )';
                            plot(stretched_demo_traj);
                        end
                        if (((nplotgroups >= 2) && (nplotgroups <= 3)) ...
                            || (nplotgroups == 6) || (nplotgroups == 14) ...
                            || (nplotgroups == 22))
                            unroll_traj = unrolling_result{np,1}{1,nplotgroups}(:,data_idx);
                            stretched_unroll_traj   = stretchTrajectory( unroll_traj', traj_stretch_length )';
                            p_unroll    = plot(stretched_unroll_traj,'g','LineWidth',3);
                        end
                        legend([p_unroll], 'unroll');
                        hold off;
                    end
                end
            end
        end
        clearvars   demo_traj stretched_demo_traj unroll_traj stretched_unroll_traj N_plot_cols
    end
%     keyboard;
end

% end of Visualizations of Nominal DMPs Fitting and Unrolling

%% Extraction of Supervised Dataset: X (input) and Ct_target (target regression variable)

dataset_Ct_tactile_asm.sub_X                        = cell(N_primitive, N_settings+1);
dataset_Ct_tactile_asm.sub_Ct_target                = cell(N_primitive, N_settings+1);
dataset_Ct_tactile_asm.sub_phase_X                  = cell(N_primitive, N_settings+1);
dataset_Ct_tactile_asm.sub_phase_V                  = cell(N_primitive, N_settings+1);
dataset_Ct_tactile_asm.sub_phase_PSI                = cell(N_primitive, N_settings+1);
baseline_unroll                                     = cell(N_primitive, N_settings+1);

smoothed_data_demo.coupled                          = cell(N_primitive, N_settings);

% Before building the dataset for each coupled settings,
% build dataset for each individual baseline demonstration against
% the (averaged) nominal behavior:
for np=1:N_primitive
    disp(['Extracting supervised dataset for setting baseline, primitive #',num2str(np),' ...']);
    N_demo                                                  = size(data_demo.baseline{np,1}, 1);
    N_modalities                                            = size(data_demo.baseline{np,1}, 2);
    dataset_Ct_tactile_asm.sub_X{np,N_settings+1}           = cell(N_demo, 1);
    dataset_Ct_tactile_asm.sub_Ct_target{np,N_settings+1}   = cell(N_demo, 1);
    dataset_Ct_tactile_asm.sub_phase_X{np,N_settings+1}     = cell(N_demo, 1);
    dataset_Ct_tactile_asm.sub_phase_V{np,N_settings+1}     = cell(N_demo, 1);
    dataset_Ct_tactile_asm.sub_phase_PSI{np,N_settings+1}   = cell(N_demo, 1);
    baseline_unroll{np,N_settings+1}                        = cell(N_demo, N_modalities);
    for ndm=1:N_demo
        % fingers
        sub_Xf_cell             = cell(1,N_finger);
        for nf=1:N_finger
            [baseline_unroll{np,N_settings+1}{ndm,BT_electrode_data_idx(1,nf)},...
             phase_X,...
             phase_V,...
             phase_PHI] = unrollBaselineSensoryPrimitiveMatchingDemo(dmp_baseline_params.BT_electrode{np,nf}, ...
                                                                     data_demo.baseline{np,1}{ndm,BT_electrode_data_idx(1,nf)}, 0);
            sub_Xf_cell{1,nf}   = data_demo.baseline{np,1}{ndm,BT_electrode_data_idx(1,nf)} - baseline_unroll{np,N_settings+1}{ndm,BT_electrode_data_idx(1,nf)};
            if (nf == 1)
                dataset_Ct_tactile_asm.sub_phase_X{np,N_settings+1}{ndm,1}  = phase_X;
                dataset_Ct_tactile_asm.sub_phase_V{np,N_settings+1}{ndm,1}  = phase_V;
                dataset_Ct_tactile_asm.sub_phase_PSI{np,N_settings+1}{ndm,1}= phase_PHI;
            end
        end
        sub_Xf_mat              = cell2mat(sub_Xf_cell);
        
        % joint sensing/proprioception
        if (is_using_joint_sensing)
            [baseline_unroll{np,N_settings+1}{ndm,22}] ...
                = unrollBaselineSensoryPrimitiveMatchingDemo(dmp_baseline_params.joint_sense{np,1}, ...
                                                          	 data_demo.baseline{np,1}{ndm,22}, 0);
            sub_Xj_mat  = data_demo.baseline{np,1}{ndm,22} - baseline_unroll{np,N_settings+1}{ndm,22};
        end
        
        dataset_Ct_tactile_asm.sub_X{np,N_settings+1}{ndm,1}= [sub_Xf_mat, sub_Xj_mat];

        % Cartesian Coordinate
        cart_coord_demo_coupled_traj_global     = cell(3,1);
        cart_coord_demo_coupled_traj_global{1,1}= data_demo.baseline{np,1}{ndm,2}(:,1:3);   % position
        cart_coord_demo_coupled_traj_global{2,1}= data_demo.baseline{np,1}{ndm,2}(:,4:6);   % velocity
        cart_coord_demo_coupled_traj_global{3,1}= data_demo.baseline{np,1}{ndm,2}(:,7:9);   % acceleration        
        
        % perform some smoothing:
        [ smoothed_cart_coord_traj ]    = smoothStartEndNDTrajectoryBasedOnPosition( cart_coord_demo_coupled_traj_global, ...
                                                                                     percentage_padding_pose_and_joint, ...
                                                                                     percentage_smoothing_points_pose_and_joint, ...
                                                                                     smoothing_mode, mean_dt, ...
                                                                                     low_pass_cutoff_freq_pose );
        cart_coord_demo_coupled_traj_global = smoothed_cart_coord_traj;
        
        [ sub_Ct_target_cart_coord_dmp ] = computeCartCoordDMPCtTarget( cart_coord_demo_coupled_traj_global,...
                                                                        dmp_baseline_params.cart_coord{np,1} );
%         [ sub_Ct_target_cart_coord_dmp ] = computeCartCoordDMPCtTargetAtNewPositionRetainOrientation( cart_coord_demo_coupled_traj_global,...
%                                                                                                       dmp_baseline_params.cart_coord{np,1} );

        % Quaternion
        Quat_demo_coupled_traj      = cell(3,1);
        Quat_demo_coupled_traj{1,1} = data_demo.baseline{np,1}{ndm,3}(:,1:4)';  % Quaternion
        Quat_demo_coupled_traj{2,1} = data_demo.baseline{np,1}{ndm,3}(:,5:7)';  % omega
        Quat_demo_coupled_traj{3,1} = data_demo.baseline{np,1}{ndm,3}(:,8:10)'; % omegad
        
        % perform some smoothing:
        [ smoothed_Quat_traj ] = smoothStartEndQuatTrajectoryBasedOnQuaternion( Quat_demo_coupled_traj, ...
                                                                                percentage_padding_pose_and_joint, ...
                                                                                percentage_smoothing_points_pose_and_joint, ...
                                                                                smoothing_mode, mean_dt, ...
                                                                                low_pass_cutoff_freq_pose );
        Quat_demo_coupled_traj      = smoothed_Quat_traj;
        
        [ sub_Ct_target_Quat_dmp ]  = computeQuatDMPCtTarget( Quat_demo_coupled_traj,...
                                                              dmp_baseline_params.Quat{np,1} );

        dataset_Ct_tactile_asm.sub_Ct_target{np,N_settings+1}{ndm,1}    = [sub_Ct_target_cart_coord_dmp, ...
                                                                           sub_Ct_target_Quat_dmp];
    end
end

% Now build the dataset for each coupled settings:
for ns=1:N_settings
    for np=1:N_primitive
        disp(['Extracting supervised dataset for setting ',num2str(ns),', primitive #',num2str(np),' ...']);
        N_demo                                      = size(data_demo.coupled{np,ns}, 1);
        N_modalities                                = size(data_demo.coupled{np,ns}, 2);
        dataset_Ct_tactile_asm.sub_X{np,ns}         = cell(N_demo, 1);
        dataset_Ct_tactile_asm.sub_Ct_target{np,ns} = cell(N_demo, 1);
        dataset_Ct_tactile_asm.sub_phase_X{np,ns}   = cell(N_demo, 1);
        dataset_Ct_tactile_asm.sub_phase_V{np,ns}   = cell(N_demo, 1);
        dataset_Ct_tactile_asm.sub_phase_PSI{np,ns} = cell(N_demo, 1);
        baseline_unroll{np,ns}                      = cell(N_demo, N_modalities);
        smoothed_data_demo.coupled{np,ns}           = cell(size(data_demo.coupled{np,ns}));
        for ndm=1:N_demo
            % fingers
            sub_Xf_cell             = cell(1,N_finger);
            for nf=1:N_finger
                [baseline_unroll{np,ns}{ndm,BT_electrode_data_idx(1,nf)},...
                 phase_X,...
                 phase_V,...
                 phase_PHI] = unrollBaselineSensoryPrimitiveMatchingDemo(dmp_baseline_params.BT_electrode{np,nf}, ...
                                                                         data_demo.coupled{np,ns}{ndm,BT_electrode_data_idx(1,nf)}, 0);
                sub_Xf_cell{1,nf}   = data_demo.coupled{np,ns}{ndm,BT_electrode_data_idx(1,nf)} - baseline_unroll{np,ns}{ndm,BT_electrode_data_idx(1,nf)};
                if (nf == 1)
                    dataset_Ct_tactile_asm.sub_phase_X{np,ns}{ndm,1}    = phase_X;
                    dataset_Ct_tactile_asm.sub_phase_V{np,ns}{ndm,1}    = phase_V;
                    dataset_Ct_tactile_asm.sub_phase_PSI{np,ns}{ndm,1}  = phase_PHI;
                end
            end
            sub_Xf_mat              = cell2mat(sub_Xf_cell);
            
            % joint sensing/proprioception
            if (is_using_joint_sensing)
                [baseline_unroll{np,ns}{ndm,22}] ...
                    = unrollBaselineSensoryPrimitiveMatchingDemo(dmp_baseline_params.joint_sense{np,1}, ...
                                                                 data_demo.coupled{np,ns}{ndm,22}, 0);
                sub_Xj_mat  = data_demo.coupled{np,ns}{ndm,22} - baseline_unroll{np,ns}{ndm,22};
            end
            
            dataset_Ct_tactile_asm.sub_X{np,ns}{ndm,1}  = [sub_Xf_mat, sub_Xj_mat];
            
            % Cartesian Coordinate
            cart_coord_demo_coupled_traj_global     = cell(3,1);
            cart_coord_demo_coupled_traj_global{1,1}= data_demo.coupled{np,ns}{ndm,2}(:,1:3);   % position
            cart_coord_demo_coupled_traj_global{2,1}= data_demo.coupled{np,ns}{ndm,2}(:,4:6);   % velocity
            cart_coord_demo_coupled_traj_global{3,1}= data_demo.coupled{np,ns}{ndm,2}(:,7:9);   % acceleration       
        
            % perform some smoothing:
            [ smoothed_cart_coord_traj ]    = smoothStartEndNDTrajectoryBasedOnPosition( cart_coord_demo_coupled_traj_global, ...
                                                                                         percentage_padding_pose_and_joint, ...
                                                                                         percentage_smoothing_points_pose_and_joint, ...
                                                                                         smoothing_mode, mean_dt, ...
                                                                                         low_pass_cutoff_freq_pose );
            cart_coord_demo_coupled_traj_global = smoothed_cart_coord_traj;
            smoothed_data_demo.coupled{np,ns}{ndm,2}    = cell2mat(cart_coord_demo_coupled_traj_global.');

            [ sub_Ct_target_cart_coord_dmp ] = computeCartCoordDMPCtTarget( cart_coord_demo_coupled_traj_global,...
                                                                            dmp_baseline_params.cart_coord{np,1} );
%             [ sub_Ct_target_cart_coord_dmp ] = computeCartCoordDMPCtTargetAtNewPositionRetainOrientation( cart_coord_demo_coupled_traj_global,...
%                                                                                                           dmp_baseline_params.cart_coord{np,1} );
            
            % Quaternion
            Quat_demo_coupled_traj      = cell(3,1);
            Quat_demo_coupled_traj{1,1} = data_demo.coupled{np,ns}{ndm,3}(:,1:4)';  % Quaternion
            Quat_demo_coupled_traj{2,1} = data_demo.coupled{np,ns}{ndm,3}(:,5:7)';  % omega
            Quat_demo_coupled_traj{3,1} = data_demo.coupled{np,ns}{ndm,3}(:,8:10)'; % omegad
        
            % perform some smoothing:
            [ smoothed_Quat_traj ] = smoothStartEndQuatTrajectoryBasedOnQuaternion( Quat_demo_coupled_traj, ...
                                                                                    percentage_padding_pose_and_joint, ...
                                                                                    percentage_smoothing_points_pose_and_joint, ...
                                                                                    smoothing_mode, mean_dt, ...
                                                                                    low_pass_cutoff_freq_pose );
            Quat_demo_coupled_traj      = smoothed_Quat_traj;
            smoothed_data_demo.coupled{np,ns}{ndm,3}    = cell2mat(Quat_demo_coupled_traj).';
        
            [ sub_Ct_target_Quat_dmp ]  = computeQuatDMPCtTarget( Quat_demo_coupled_traj,...
                                                                  dmp_baseline_params.Quat{np,1} );
            
            dataset_Ct_tactile_asm.sub_Ct_target{np,ns}{ndm,1}  = [sub_Ct_target_cart_coord_dmp, ...
                                                                   sub_Ct_target_Quat_dmp];
        end
    end
end
save(['dataset_Ct_tactile_asm_',task_type,'.mat'],'dataset_Ct_tactile_asm');
save(['smoothed_data_demo_',task_type,'.mat'],'smoothed_data_demo');

% end of Extraction of Supervised Dataset: X (input) and Ct_target (target regression variable)