function [ X_gT, Yo, Ydo, Yddo, L_x3, L_v3, Ct ] = constructSynthObsAvoidCircleTraj( w_DMP, w_Ct_SYNTH, start, goal, obs, traj_length, dt, beta_grid, k_grid, c_order, obs_radius )
    global dcps;
    n_rfs           = size(w_DMP,1);
    tau             = (traj_length-1)*dt;

    Yo              = zeros(traj_length,2);
    Ydo             = zeros(traj_length,2);
    Yddo            = zeros(traj_length,2);
    Ct              = zeros(traj_length,2);

    o3              = zeros(3,1);
    x3              = zeros(3,1);
    v3              = zeros(3,1);

    X_gT            = zeros(traj_length, 2*length(beta_grid));

    % latent variable logging:
    L_x3            = zeros(traj_length,2);
    L_v3            = zeros(traj_length,2);

    % initialize x3 and v3:
    o3(1:2,1)       = obs;
    x3(1:2,1)       = start(1,:)';
    v3(1:2,1)       = zeros(2,1);

    for d=1:2
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d, start(1,d));
        dcp_franzi('set_goal',d,goal(1,d),1);

        dcps(d).w   = w_DMP(:,d);
    end

    for i=1:traj_length
        L_x3(i,:)       = x3(1:2,1)';
        L_v3(i,:)       = v3(1:2,1)';

        % x3 and v3 computed here is the "ground truth".
        % compute model-based coupling term:
        for j=1:length(beta_grid)
            ct3         = computeAksharaHumanoids2014ObstAvoidCtSphere( beta_grid(j,1), k_grid(j,1), x3, v3, o3, obs_radius );
            X_gT(i,((j-1)*2+1):((j-1)*2+2)) = ct3(1:2,1)';
        end
        ct              = X_gT(i,:) * w_Ct_SYNTH;
        Ct(i,:)         = ct;
        for d=1:2
            [y,yd,ydd,f] = dcp_franzi('run',d,tau,dt,ct(1,d));

            Yo(i,d)     = y;
            Ydo(i,d)    = yd;
            Yddo(i,d)   = ydd;
        end

        x3(1:2,1)       = Yo(i,:)';
        v3(1:2,1)       = Ydo(i,:)';
    end
end
