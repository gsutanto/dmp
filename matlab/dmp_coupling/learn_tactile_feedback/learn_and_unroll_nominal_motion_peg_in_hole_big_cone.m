% Author: Giovanni Sutanto
% Date  : February 1, 2017

clear  	all;
close   all;
clc;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);

task_type           = 'peg_in_hole_big_cone/baseline/';
in_data_dir_path    = [rel_dir_path...
                       ,'../../../data/dmp_coupling/learn_tactile_feedback/'...
                       ,task_type,'/'];

%% Data Loading

[ data_demo ]   = extractSensoriMotorTracesSingleSetting( in_data_dir_path );

N_primitive     = size(data_demo,1);

%% Fitting DMPs and Unrolling

dmp_weights         = cell(N_primitive,1);
unrolling_result    = cell(N_primitive,1);

n_rfs               = 25;
c_order             = 1;

traj_stretch_length = 1500;

for np=1:N_primitive
    dmp_weights{np,1}       = cell(1,size(data_demo{np,1},2));
    unrolling_result{np,1}  = cell(1,size(data_demo{np,1},2));
    N_traj                  = size(data_demo{np,1},1);
    
    %% Cartesian Coordinate DMP Fitting and Unrolling
    
    % Load Cartesian Coordinate Trajectory Demonstrations into Structure:
    cart_coord_traj_demo_set    = cell(3, N_traj);
    dts                         = zeros(1, N_traj);
    for nt=1:N_traj
        time        = data_demo{np,1}{nt,1}';
        XT          = data_demo{np,1}{nt,2}(:,1:3);
        XdT         = data_demo{np,1}{nt,2}(:,4:6);
        XddT        = data_demo{np,1}{nt,2}(:,7:9);

        tau         = time(1,end) - time(1,1);
        traj_length = size(XT,1);
        dt          = tau/(traj_length-1);
    
        cart_coord_traj_demo_set{1, nt} = XT;
        cart_coord_traj_demo_set{2, nt} = XdT;
        cart_coord_traj_demo_set{3, nt} = XddT;
        dts(1,nt)   = dt;
    end
    clearvars       time tau traj_length dt XT XdT XddT
    
    mean_dt = mean(dts);
    
    disp('Processing Local Coordinate System for Demonstrated Cartesian Coordinate Trajectories and Fitting its Primitive ...');
    [ cart_coord_dmp_baseline_params, ...
      cart_coord_dmp_baseline_unroll_global_traj ] = learnCartPrimitiveMultiOnLocalCoord(cart_coord_traj_demo_set, mean_dt, n_rfs, c_order, 1);
  
    dmp_weights{np,1}{1,2}      = cart_coord_dmp_baseline_params.w;
    unrolling_result{np,1}{1,2} = [cart_coord_dmp_baseline_unroll_global_traj{1,1},...
                                   cart_coord_dmp_baseline_unroll_global_traj{2,1},...
                                   cart_coord_dmp_baseline_unroll_global_traj{3,1}];
    
    %% Quaternion DMP Fitting and Unrolling
    
    % Load Quaternion Trajectory Demonstrations into Structure:
    Quat_traj_demo_set  = cell(3, N_traj);
    for nt=1:N_traj
        QT          = data_demo{np,1}{nt,3}(:,1:4)';
        omegaT      = data_demo{np,1}{nt,3}(:,5:7)';
        omegadT     = data_demo{np,1}{nt,3}(:,8:10)';

        Quat_traj_demo_set{1, nt} = QT;
        Quat_traj_demo_set{2, nt} = omegaT;
        Quat_traj_demo_set{3, nt} = omegadT;
    end
    clearvars       QT omegaT omegadT;
    
    [ Quat_dmp_baseline_params, ...
      Quat_dmp_baseline_unroll_traj ]   = learnQuatPrimitiveMulti(Quat_traj_demo_set, ...
                                                                  mean_dt, ...
                                                                  n_rfs, ...
                                                                  c_order, ...
                                                                  cart_coord_dmp_baseline_params.mean_tau);
  
    dmp_weights{np,1}{1,3}      = Quat_dmp_baseline_params.w;
    unrolling_result{np,1}{1,3} = [Quat_dmp_baseline_unroll_traj{1,1},...
                                   Quat_dmp_baseline_unroll_traj{4,1},...
                                   Quat_dmp_baseline_unroll_traj{5,1}];
    
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
end

%% Visualizations

for np=1:N_primitive
    N_traj          = size(data_demo{np,1},1);
    
    % The columns of plot_titles and plot_indices below corresponds to 
    % the columns of data_demo{np,1}.
    % plot_indices represents the range of data plotting 
    % (leftmost column of data_demo{np,1}{nt,nplotgroups})
    % plot_indices [0] means data_demo{np,1}{:,nplotgroups} will not be
    % plotted:
    plot_titles     = { {}, {'Cartesian Coordinate X','Cartesian Coordinate Xd','Cartesian Coordinate Xdd'} ...
                       ,{'Cartesian Quaternion Q','Cartesian Quaternion omega','Cartesian Quaternion omegad'} ...
                       ,{'R\_HAND FT Force'}, {'R\_HAND FT Torque'} ...
                       ,{'R\_LF\_electrodes'}, {'R\_LF\_TDC'} ...
%                        ,{'R\_LF\_TAC'}, {'R\_LF\_PDC'}, {'R\_LF\_PACs'}...
%                        ,{},{},{'R\_LF computed 3D POC'}...
%                        ,{'R\_RF\_electrodes'}, {'R\_RF\_TDC'} ...
%                        ,{'R\_RF\_TAC'}, {'R\_RF\_PDC'}, {'R\_RF\_PACs'}...
%                        ,{},{},{'R\_RF computed 3D POC'}...
                       };
    plot_indices    = { {},{[1:3],[4:6],[7:9]}...
                       ,{[1:4],[5:7],[8:10]}...
                       ,{[1:3]},{[1:3]}...
                       ,{[1:5]},{[1]}...
%                        ,{[1]},{[1]},{[1:5]}...
%                        ,{},{},{[1:3]}...
%                        ,{[1:5]},{[1]}...
%                        ,{[1]},{[1]},{[1:5]}...
%                        ,{},{},{[1:3]}...
                       };
                   
    for nplotgroups=1:size(plot_indices,2)
        if (~isempty(plot_indices{1,nplotgroups}))
            for nplot=1:size(plot_indices{1,nplotgroups},2)
                figure;
                D   = plot_indices{1,nplotgroups}{1,nplot}(1,end)-(plot_indices{1,nplotgroups}{1,nplot}(1,1)-1);
                for d=1:D
                    data_idx    = (plot_indices{1,nplotgroups}{1,nplot}(1,1)-1) + d;
                    subplot(D,1,d);
                    if (d==1)
                        title([plot_titles{1,nplotgroups}{1,nplot},', Primitive #',num2str(np)]);
                    end
                    hold on;
                    for nt=1:N_traj
                        demo_traj= data_demo{np,1}{nt,nplotgroups}(:,data_idx);
                        stretched_demo_traj = stretchTrajectory( demo_traj', traj_stretch_length )';
                        plot(stretched_demo_traj);
                    end
                    if ((nplotgroups >= 2) && (nplotgroups <= 3))   % cartesian coordinate or Quaternion
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
end