% Author: Giovanni Sutanto
% Date  : July 2017

clear  	all;
close   all;
clc;

addpath('../../utilities/');
addpath('../../cart_dmp/cart_coord_dmp/');
addpath('../../cart_dmp/quat_dmp/');
addpath('../../dmp_multi_dim/');

specific_task_type  = 'scraping_w_tool';
in_data_dir_path    = ['../../../data/dmp_coupling/learn_tactile_feedback/'...
                       ,specific_task_type,'/'];
is_visualize_baseline_data_fitting  = 1;
                   
%% Data Loading

[ data_demo_human_baseline ]   = extractNominalSensoriMotorTraces( in_data_dir_path );
save(['data_demo_human_baseline_',specific_task_type,'.mat'],'data_demo_human_baseline');

N_primitive     = size(data_demo_human_baseline, 1);

% end of Data Loading

%% Fitting Nominal DMPs and Unrolling

action_dmp_baseline_params.cart_coord   = cell(N_primitive, 1);
action_dmp_baseline_params.Quat         = cell(N_primitive, 1);
unrolling_result                        = cell(N_primitive, 1);

smoothed_data_demo_human_baseline.baseline     = cell(N_primitive, 1);

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
    
    unrolling_result{np,1}      = cell(1,size(data_demo_human_baseline{np,1},2));
    N_traj                      = size(data_demo_human_baseline{np,1},1);
    
    %% Cartesian Coordinate DMP Fitting and Unrolling
    
    % Load Cartesian Coordinate Trajectory Demonstrations into Structure:
    cart_coord_traj_demo_set    = cell(3, N_traj);
    dts                         = zeros(1, N_traj);
    smoothed_data_demo_human_baseline.baseline{np,1}   = cell(size(data_demo_human_baseline{np,1}));
    for nt=1:N_traj
        time        = data_demo_human_baseline{np,1}{nt,1}';
        XT          = data_demo_human_baseline{np,1}{nt,2}(:,1:3);
        XdT         = data_demo_human_baseline{np,1}{nt,2}(:,4:6);
        XddT        = data_demo_human_baseline{np,1}{nt,2}(:,7:9);

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
        smoothed_data_demo_human_baseline.baseline{np,1}{nt,2} = cell2mat(cart_coord_traj_demo_set(:,nt).');
        
        dts(1,nt)   = dt;
    end
    clearvars       time tau traj_length dt XT XdT XddT;
    
    assert(var(dts) < 1e-10, 'Sampling Time (dt) is inconsistent across demonstrated trajectories.');
    mean_dt         = mean(dts);
    cart_coord_dmp_unroll_dt            = mean_dt;
    cart_coord_dmp_is_using_scaling     = [1, 0, 0];
    
    [ cart_coord_action_dmp_baseline_params, ...
      cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(cart_coord_traj_demo_set, ...
                                                                                         mean_dt, n_rfs, c_order, ...
                                                                                         ctraj_local_coordinate_frame_selection, ...
                                                                                         -1, cart_coord_dmp_unroll_dt, ...
                                                                                         cart_coord_dmp_is_using_scaling);
  
    action_dmp_baseline_params.cart_coord{np,1}    = cart_coord_action_dmp_baseline_params;
    unrolling_result{np,1}{1,2} = [cart_coord_dmp_baseline_unroll_global_traj{1,1},...
                                   cart_coord_dmp_baseline_unroll_global_traj{2,1},...
                                   cart_coord_dmp_baseline_unroll_global_traj{3,1}];
    
    % end of Cartesian Coordinate DMP Fitting and Unrolling
    
    %% Quaternion DMP Fitting and Unrolling
    
    % Load Quaternion Trajectory Demonstrations into Structure:
    Quat_traj_demo_set  = cell(3, N_traj);
    for nt=1:N_traj
        QT          = data_demo_human_baseline{np,1}{nt,3}(:,1:4)';
        omegaT      = data_demo_human_baseline{np,1}{nt,3}(:,5:7)';
        omegadT     = data_demo_human_baseline{np,1}{nt,3}(:,8:10)';

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
        smoothed_data_demo_human_baseline.baseline{np,1}{nt,3} = cell2mat(Quat_traj_demo_set(:,nt)).';
    end
    clearvars       QT omegaT omegadT;
    
    [ Quat_action_dmp_baseline_params, ...
      Quat_dmp_baseline_unroll_traj ]   = learnQuatPrimitiveMulti(Quat_traj_demo_set, ...
                                                                  mean_dt, ...
                                                                  n_rfs, ...
                                                                  c_order, ...
                                                                  cart_coord_action_dmp_baseline_params.mean_tau);
  
    action_dmp_baseline_params.Quat{np,1}  = Quat_action_dmp_baseline_params;
    unrolling_result{np,1}{1,3} = [Quat_dmp_baseline_unroll_traj{1,1},...
                                   Quat_dmp_baseline_unroll_traj{4,1},...
                                   Quat_dmp_baseline_unroll_traj{5,1}];
    
    % end of Quaternion DMP Fitting and Unrolling
    
    %% Logging Unrolling Results into Files
    
    unroll_cart_coord_dmp_result_dump_dir_path  = [pwd,'/unroll/',specific_task_type...
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
         
    unroll_quat_dmp_result_dump_dir_path    = [pwd,'/unroll/',specific_task_type...
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
    clearvars       cart_coord_traj_demo_set cart_coord_action_dmp_baseline_params cart_coord_dmp_baseline_unroll_global_traj dts;
    clearvars       Quat_traj_demo_set Quat_action_dmp_baseline_params Quat_dmp_baseline_unroll_traj;
    clearvars       unroll_cart_coord_dmp_result_dump unroll_quat_dmp_result_dump
    
    % end of Logging Unrolling Results into Files
end

save(['action_dmp_baseline_params_',specific_task_type,'.mat'],'action_dmp_baseline_params');
save(['smoothed_data_demo_human_baseline_',specific_task_type,'.mat'],'smoothed_data_demo_human_baseline');

% end of Fitting Nominal DMPs and Unrolling

%% Visualizations of Nominal DMPs Fitting and Unrolling

if (is_visualize_baseline_data_fitting)
    for np=1:N_primitive
        N_traj          = size(data_demo_human_baseline{np,1},1);

        % The columns of plot_titles and plot_indices below corresponds to 
        % the columns of data_demo_human_baseline.baseline{np,1}.
        % plot_indices represents the range of data plotting 
        % (leftmost column of data_demo_human_baseline.baseline{np,1}{nt,nplotgroups})
        % plot_indices [0] means data_demo_human_baseline.baseline{np,1}{:,nplotgroups} will not be
        % plotted:
        plot_titles     = { {}, {'Cartesian Coordinate X','Cartesian Coordinate Xd','Cartesian Coordinate Xdd'} ...
                           ,{'Cartesian Quaternion Q','Cartesian Quaternion omega','Cartesian Quaternion omegad'} ...
                           ,{}, {} ...%,{'R\_HAND FT Force'}, {'R\_HAND FT Torque'} ...
                           ,{} ...%,{'R\_LF\_electrodes'} ...
                           ,{} ...%,{'R\_LF\_TDC'} ...
                           ,{}, {}, {} ...%,{'R\_LF\_TAC'}, {'R\_LF\_PDC'}, {'R\_LF\_PACs'}...
                           ,{},{} ...
                           ,{} ...%,{'R\_LF computed 3D POC'}...
                           ,{} ...%,{'R\_RF\_electrodes'}...
                           ,{} ...%, {'R\_RF\_TDC'} ...
                           ,{}, {}, {} ...%,{'R\_RF\_TAC'}, {'R\_RF\_PDC'}, {'R\_RF\_PACs'}...
                           ,{},{} ...
                           ,{} ...%,{'R\_RF computed 3D POC'}...
                           };
        plot_indices    = { {},{[1:3],[4:6],[7:9]}...
                           ,{[1:4],[5:7],[8:10]}...
                           ,{}, {} ...%,{[1:3]},{[1:3]}...
                           ,{} ...%,{[1:19]}...
                           ,{} ...%,{[1]}...
                           ,{}, {}, {} ...%,{[1]},{[1]},{[1:5]}...
                           ,{},{} ...
                           ,{} ...%,{[1:3]}...
                           ,{} ...%,{[1:19]}...
                           ,{} ...%,{[1]}...
                           ,{}, {}, {} ...%,{[1]},{[1]},{[1:5]}...
                           ,{},{} ...
                           ,{} ...%,{[1:3]}...
                           };

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
                            demo_traj= data_demo_human_baseline{np,1}{nt,nplotgroups}(:,data_idx);
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