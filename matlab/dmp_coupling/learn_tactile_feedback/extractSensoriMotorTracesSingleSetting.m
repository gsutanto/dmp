function [ data_demo ] = extractSensoriMotorTracesSingleSetting( in_data_dir_path )
    exclude_file_path   = [in_data_dir_path, '/exclude.txt'];
    if (exist(exclude_file_path, 'file'))
        exclude_indices = dlmread(exclude_file_path);
    else
        exclude_indices = [];
    end
    
    N_primitive         = 1;    % # of primitives
    prim_trajs_path     = [in_data_dir_path, 'prim', num2str(N_primitive,'%02d'),'/'];
    while (exist(prim_trajs_path, 'dir') == 7)
        N_primitive     = N_primitive + 1;
        prim_trajs_path = [in_data_dir_path, 'prim', num2str(N_primitive,'%02d'),'/'];
    end
    N_primitive         = N_primitive - 1;
    
    data_demo           = cell(N_primitive,1);
    
    for prim_count=1:N_primitive
        prim_trajs_path = [in_data_dir_path, 'prim', num2str(prim_count,'%02d'),'/'];
        data_demo{prim_count,1} = cell(0);  % each column corresponds to a primitive
        file_count      = 1;
        prim_traj_file_path = [prim_trajs_path, num2str(file_count,'%02d'),'.txt'];
        traj_count      = 1;
        while (exist(prim_traj_file_path, 'file'))
            if (any(file_count==exclude_indices)==0)
                sm_traj     = dlmread(prim_traj_file_path);

                % each row corresponds to a demo; 
                % each column corresponds to a sensorimotor modality trajectory:
                data_demo{prim_count,1}{traj_count,1}   = sm_traj(:,1);         % time
                data_demo{prim_count,1}{traj_count,2}   = sm_traj(:,2:10);      % cartesian coordinate trajectory (x,y,z,xd,yd,zd,xdd,ydd,zdd)
                data_demo{prim_count,1}{traj_count,3}   = sm_traj(:,11:20);     % cartesian Quaternion trajectory (q0,q1,q2,q3,ad,bd,gd,add,bdd,gdd)
                data_demo{prim_count,1}{traj_count,4}   = sm_traj(:,21:23);     % R_HAND Force Sensor (of Force-Torque Sensor)
                data_demo{prim_count,1}{traj_count,5}   = sm_traj(:,24:26);     % R_HAND Torque Sensor (of Force-Torque Sensor)
                data_demo{prim_count,1}{traj_count,6}   = sm_traj(:,27:45);     % BioTac R_LF_electrodes
                data_demo{prim_count,1}{traj_count,7}   = sm_traj(:,46);        % BioTac R_LF_TDC
                data_demo{prim_count,1}{traj_count,8}   = sm_traj(:,47);        % BioTac R_LF_TAC
                data_demo{prim_count,1}{traj_count,9}   = sm_traj(:,48);        % BioTac R_LF_PDC
                data_demo{prim_count,1}{traj_count,10}  = sm_traj(:,49:70);     % BioTac R_LF_PACs (22 data points)
                data_demo{prim_count,1}{traj_count,11}  = sm_traj(:,71:73);     % BioTac R_LF computed 3D Force     (computed from electrodes data)
                data_demo{prim_count,1}{traj_count,12}  = sm_traj(:,74:76);     % BioTac R_LF computed 3D Torque    (computed from electrodes data)
                data_demo{prim_count,1}{traj_count,13}  = sm_traj(:,77:79);     % BioTac R_LF computed 3D POC       (computed from electrodes data)
                data_demo{prim_count,1}{traj_count,14}  = sm_traj(:,94:112);    % BioTac R_RF_electrodes
                data_demo{prim_count,1}{traj_count,15}  = sm_traj(:,113);       % BioTac R_RF_TDC
                data_demo{prim_count,1}{traj_count,16}  = sm_traj(:,114);       % BioTac R_RF_TAC
                data_demo{prim_count,1}{traj_count,17}  = sm_traj(:,115);       % BioTac R_RF_PDC
                data_demo{prim_count,1}{traj_count,18}  = sm_traj(:,116:137);   % BioTac R_RF_PACs (22 data points)
%                 data_demo{prim_count,1}{traj_count,19}  = sm_traj(:,138:140);   % BioTac R_RF computed 3D Force     (computed from electrodes data)
%                 data_demo{prim_count,1}{traj_count,20}  = sm_traj(:,141:143);   % BioTac R_RF computed 3D Torque    (computed from electrodes data)
%                 data_demo{prim_count,1}{traj_count,21}  = sm_traj(:,144:146);   % BioTac R_RF computed 3D POC       (computed from electrodes data)
                data_demo{prim_count,1}{traj_count,19}  = sm_traj(:,80:82);     % R_HAND Force Sensor (of Force-Torque Sensor) w.r.t. World/Global Coordinate System
                data_demo{prim_count,1}{traj_count,20}  = sm_traj(:,87:89);     % R_HAND Torque Sensor (of Force-Torque Sensor) w.r.t. World/Global Coordinate System
                data_demo{prim_count,1}{traj_count,21}  = sm_traj(:,83:86);     % PI gating gains for z-axis Force Control and y-axis Torque Control based on Force-Torque Sensor
                if (size(sm_traj,2) > 146)
                    data_demo{prim_count,1}{traj_count,22}= sm_traj(:,147:153); % Joint Positions/Coordinates
                end
                if (size(sm_traj,2) > 153)
                    data_demo{prim_count,1}{traj_count,23}= sm_traj(:,154:159); % Coupling Terms (6-D: 3-D position x-y-z and 3-D orientation alpha-beta-gamma, output of PMNN)
                end
                if (size(sm_traj,2) > 159)
                    data_demo{prim_count,1}{traj_count,24}= sm_traj(:,160:165); % Supposed (Computed but might be NOT Applied) Coupling Terms (6-D: 3-D position x-y-z and 3-D orientation alpha-beta-gamma, output of PMNN)
                end
                if (size(sm_traj,2) > 165)
                    data_demo{prim_count,1}{traj_count,25}= sm_traj(:,166:210); % X_vector (Delta X, the feature vector input for PMNN)
                end

                traj_count  = traj_count + 1;
            end

            file_count  = file_count + 1;
            prim_traj_file_path = [prim_trajs_path, num2str(file_count,'%02d'),'.txt'];
        end
    end
end