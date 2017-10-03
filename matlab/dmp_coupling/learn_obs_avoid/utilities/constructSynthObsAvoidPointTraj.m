function [ X_latent, Yo, Ydo, Yddo, L_ox3, L_v3, Ct ] = constructSynthObsAvoidPointTraj( w_DMP, w_Ct_SYNTH, start, goal, obs, traj_length, dt, beta_grid, k_grid, c_order )
    global dcps;
    n_rfs           = size(w_DMP,1);
    tau             = (traj_length-1)*dt;

    Yo              = zeros(traj_length,2);
    Ydo             = zeros(traj_length,2);
    Yddo            = zeros(traj_length,2);
    Ct              = zeros(traj_length,2);

    ox3             = zeros(3,1);
    v3              = zeros(3,1);

    X_latent        = zeros(traj_length, 2*length(beta_grid));

    % latent variable logging:
    L_ox3           = zeros(traj_length,2);
    L_v3            = zeros(traj_length,2);

    % initialize ox3 and v3:
    ox3(1:2,1)      = obs-start(1,:)';
    v3(1:2,1)       = zeros(2,1);

    for d=1:2
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d, start(1,d));
        dcp_franzi('set_goal',d,goal(1,d),1);

        dcps(d).w   = w_DMP(:,d);
    end

    for i=1:traj_length
        L_ox3(i,:)      = ox3(1:2,1)';
        L_v3(i,:)       = v3(1:2,1)';

        % ox3 and v3 computed here is the "ground truth".
        % compute model-based coupling term:
        for j=1:length(beta_grid)
            ct3         = computeAksharaHumanoids2014ObstAvoidCtPoint( beta_grid(j,1), k_grid(j,1), ox3, v3 );
            X_latent(i,((j-1)*2+1):((j-1)*2+2)) = ct3(1:2,1)';
        end
        ct              = X_latent(i,:) * w_Ct_SYNTH;
        Ct(i,:)         = ct;
        for d=1:2
            [y,yd,ydd,f] = dcp_franzi('run',d,tau,dt,ct(1,d));

            Yo(i,d)     = y;
            Ydo(i,d)    = yd;
            Yddo(i,d)   = ydd;
        end

        ox3(1:2,1)      = obs-Yo(i,:)';
        v3(1:2,1)       = Ydo(i,:)';
    end
end
