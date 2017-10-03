% Author: Giovanni Sutanto
% Date  : February 1, 2017

clear  	all;
close   all;
clc;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);

demo_type           = 'unscrewing';
in_data_dir_path    = [rel_dir_path...
                       ,'../../../data/dmp_coupling/learn_tactile_feedback/'...
                       ,demo_type,'/'];

%% Data Loading

data_demo   = cell(0);

prim_count          = 1;    % primitive count
prim_trajs_path     = [in_data_dir_path, 'prim', num2str(prim_count,'%02d'),'/'];
while (exist(prim_trajs_path, 'dir') == 7)
    data_demo{prim_count,1} = cell(0);  % each column corresponds to a primitive
    traj_count      = 1;
    prim_traj_file_path = [prim_trajs_path, num2str(traj_count,'%02d'),'.txt'];
    while (exist(prim_traj_file_path, 'file'))
        sm_traj     = dlmread(prim_traj_file_path);
        
        % each row corresponds to a demo; 
        % each column corresponds to a sensorimotor modality trajectory:
        data_demo{prim_count,1}{traj_count,1}   = sm_traj(:,1);     % time
        data_demo{prim_count,1}{traj_count,2}   = sm_traj(:,2:10);  % cartesian coordinate trajectory (x,y,z,xd,yd,zd,xdd,ydd,zdd)
        data_demo{prim_count,1}{traj_count,3}   = sm_traj(:,11:20); % cartesian Quaternion trajectory (q0,q1,q2,q3,ad,bd,gd,add,bdd,gdd)
        data_demo{prim_count,1}{traj_count,4}   = sm_traj(:,21:23); % R_HAND Force Sensor (of Force-Torque Sensor)
        data_demo{prim_count,1}{traj_count,5}   = sm_traj(:,24:26); % R_HAND Torque Sensor (of Force-Torque Sensor)
        data_demo{prim_count,1}{traj_count,6}   = sm_traj(:,27:45); % BioTac R_LF_electrodes
        data_demo{prim_count,1}{traj_count,7}   = sm_traj(:,46);    % BioTac R_LF_TDC
        data_demo{prim_count,1}{traj_count,8}   = sm_traj(:,47);    % BioTac R_LF_TAC
        data_demo{prim_count,1}{traj_count,9}   = sm_traj(:,48);    % BioTac R_LF_PDC
        data_demo{prim_count,1}{traj_count,10}  = sm_traj(:,49:70); % BioTac R_LF_PACs (22 data points)
        data_demo{prim_count,1}{traj_count,11}  = sm_traj(:,71:73); % BioTac R_LF computed 3D Force     (computed from electrodes data)
        data_demo{prim_count,1}{traj_count,12}  = sm_traj(:,74:76); % BioTac R_LF computed 3D Torque    (computed from electrodes data)
        data_demo{prim_count,1}{traj_count,13}  = sm_traj(:,77:79); % BioTac R_LF computed 3D POC       (computed from electrodes data)
        
        traj_count  = traj_count + 1;
        prim_traj_file_path = [prim_trajs_path, num2str(traj_count,'%02d'),'.txt'];
    end
    prim_count      = prim_count + 1;
    prim_trajs_path = [in_data_dir_path, 'prim', num2str(prim_count,'%02d'),'/'];
end
N_primitive         = prim_count - 1;

%% Fitting DMPs and Unrolling

dmp_weights         = cell(N_primitive,1);
unrolling_result    = cell(N_primitive,1);

global              dcps;

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
    Q0s                 = zeros(4, N_traj);
    QGs                 = zeros(4, N_traj);
    for nt=1:N_traj
        QT          = data_demo{np,1}{nt,3}(:,1:4)';
        omegaT      = data_demo{np,1}{nt,3}(:,5:7)';
        omegadT     = data_demo{np,1}{nt,3}(:,8:10)';

        % Initial Orientation Quaternion:
        Q0      = QT(:,1);
        Q0      = Q0/norm(Q0);

        % Goal Orientation Quaternion:
        QG      = QT(:,end);
        QG      = QG/norm(QG);

        Quat_traj_demo_set{1, nt} = QT;
        Quat_traj_demo_set{2, nt} = omegaT;
        Quat_traj_demo_set{3, nt} = omegadT;
        Q0s(:,nt)   = Q0;
        QGs(:,nt)   = QG;
    end
    clearvars       QT omegaT omegadT Q0 QG;
    
    assert(var(dts) < 1e-10, 'Sampling Time (dt) is inconsistent across demonstrated trajectories.');
    if (isQuatWithNegativeRealParts(Q0s))
        mean_Q0 = -computeAverageQuaternions(Q0s);
    else
        mean_Q0 = computeAverageQuaternions(Q0s);
    end
    if (isQuatWithNegativeRealParts(QGs))
        mean_QG = -computeAverageQuaternions(QGs);
    else
        mean_QG = computeAverageQuaternions(QGs);
    end

    ID     	= 1;

    % Fitting/Learning the Quaternion DMP based on Dataset
    disp('Fitting/Learning the Quaternion DMP based on Dataset ...');
    dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
    dcp_quaternion('reset_state', ID, mean_Q0);
    dcp_quaternion('set_goal', ID, mean_QG, 1);

    [w_dmp_quat, F_target, F_fit]   = dcp_quaternion('batch_fit_multi', ID, mean_dt, Quat_traj_demo_set);
    dmp_weights{np,1}{1,3}          = w_dmp_quat;
    
    mean_tau        = cart_coord_dmp_baseline_params.mean_tau;
    mean_traj_length= round(mean_tau/mean_dt) + 1;
    
    % Unrolling based on Dataset (using mean_Q0 and mean_QG)
    time_unroll     = zeros(1, mean_traj_length);
    Q_unroll        = zeros(4, mean_traj_length);
    Qd_unroll       = zeros(4, mean_traj_length);
    Qdd_unroll      = zeros(4, mean_traj_length);
    omega_unroll    = zeros(3, mean_traj_length);
    omegad_unroll   = zeros(3, mean_traj_length);
    F_run           = zeros(3, mean_traj_length);

    dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
    dcp_quaternion('reset_state', ID, mean_Q0);
    dcp_quaternion('set_goal', ID, mean_QG, 1);
    dcps(1).w       = dmp_weights{np,1}{1,3};

    % t_unroll==0 corresponds to initial conditions
    t_unroll        = 0;
    for i=1:mean_traj_length
        [Q, Qd, Qdd, omega, omegad, f] = dcp_quaternion('run', ID, mean_tau, mean_dt);
        t_unroll            = t_unroll + mean_dt;

        time_unroll(:,i)    = t_unroll;
        
        Q_unroll(:,i)       = Q;
        Qd_unroll(:,i)      = Qd;
        Qdd_unroll(:,i)     = Qdd;
        omega_unroll(:,i)   = omega;
        omegad_unroll(:,i)  = omegad;

        F_run(:,i)          = f;
    end
    
    unrolling_result{np,1}{1,3} = [Q_unroll', omega_unroll', omegad_unroll'];
    
    %% Logging Unrolling Results into Files
    
    unroll_cart_coord_dmp_result_dump_file_path = [pwd,'/unroll/',demo_type...
                                                   ,'/cart_dmp/cart_coord_dmp/'...
                                                   ,num2str(np),'.txt'];
    unroll_cart_coord_dmp_result_dump = [  cart_coord_dmp_baseline_unroll_global_traj{4,1}...
                                         , cart_coord_dmp_baseline_unroll_global_traj{1,1}...
                                         , cart_coord_dmp_baseline_unroll_global_traj{2,1}...
                                         , cart_coord_dmp_baseline_unroll_global_traj{3,1}];
    dlmwrite(unroll_cart_coord_dmp_result_dump_file_path, unroll_cart_coord_dmp_result_dump, ...
             'delimiter', ' ');
         
    unroll_quat_dmp_result_dump_file_path   = [pwd,'/unroll/',demo_type...
                                               ,'/cart_dmp/quat_dmp/'...
                                               ,num2str(np),'.txt'];
    unroll_quat_dmp_result_dump = [time_unroll'...
                                   , Q_unroll', Qd_unroll', Qdd_unroll'...
                                   , omega_unroll', omegad_unroll'];
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
%                        ,{'R\_HAND FT Force'}, {'R\_HAND FT Torque'} ...
%                        ,{'R\_LF\_electrodes'}, {'R\_LF\_TDC'} ...
%                        ,{'R\_LF\_TAC'}, {'R\_LF\_PDC'}, {'R\_LF\_PACs'},...
%                        ,{},{},{'R\_LF computed 3D POC'}...
                       };
    plot_indices    = { {},{[1:3],[4:6],[7:9]}...
                       ,{[1:4],[5:7],[8:10]}...
%                        ,{[1:3]},{[1:3]}...
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