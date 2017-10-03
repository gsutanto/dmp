function [ X, Fo, Yo, Yod, Yodd, T_ox3, T_v3 ] = constructObsAvoidPointFeatMat2D_old( w_obs_DMP, start, goal, obs, traj_length, dt, beta_grid, k_grid, c_order )
    global dcps;
    n_rfs           = size(w_obs_DMP,1);
    
    for d=1:2
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d,start(1,d));
        dcp_franzi('set_goal',d,goal(1,d),1);

        dcps(d).w   = w_obs_DMP(:,d);
    end
    
    tau             = (traj_length-1)*dt;
    
    Yo              = zeros(traj_length,2);
    Yod             = zeros(traj_length,2);
    Yodd            = zeros(traj_length,2);
    Fo              = zeros(traj_length,2);

    ox3             = zeros(3,1);
    v3              = zeros(3,1);

    % Observed variable logging:
    T_ox3           = zeros(traj_length,2);
    T_v3            = zeros(traj_length,2);

    X               = zeros(traj_length, 2*length(beta_grid));

    % First unroll the obstacle avoidance trajectory:
    for i=1:traj_length
        for d=1:2
            [y,yd,ydd,f] = dcp_franzi('run',d,tau,dt);

            Yo(i,d)   = y;
            Yod(i,d)  = yd;
            Yodd(i,d) = ydd;
            Fo(i,d)   = f;
        end

        ox3(1:2,1)      = obs-Yo(i,:)';
        v3(1:2,1)       = Yod(i,:)';

        T_ox3(i,:)      = ox3(1:2,1)';
        T_v3(i,:)       = v3(1:2,1)';

        % ox3 and v3 computed here is the "observed" variable (in HMM sense).

        for j=1:length(beta_grid)
            ct3         = computeAksharaHumanoids2014ObstAvoidCtPoint( beta_grid(j,1), k_grid(j,1), ox3, v3 );
            X(i,((j-1)*2+1):((j-1)*2+2)) = ct3(1:2,1)';
        end
    end
end
