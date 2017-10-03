function [ Y, Yd, Ydd, F ] = unrollObsAvoidPointTraj( w_DMP, w_Ct, start, goal, obs, traj_length, dt, beta_grid, k_grid, c_order )
    global dcps;
    n_rfs           = size(w_DMP,1);
    
    for d=1:2
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d,start(1,d));
        dcp_franzi('set_goal',d,goal(1,d),1);

        dcps(d).w   = w_DMP(:,d);
    end
    
    tau             = (traj_length-1)*dt;

    Y               = zeros(traj_length,2);
    Yd              = zeros(traj_length,2);
    Ydd             = zeros(traj_length,2);
    F               = zeros(traj_length,2);
    ct3             = zeros(3,1);
    ct              = zeros(1,2);
    Ct              = zeros(traj_length,2);

    ox3             = zeros(3,1);
    v3              = zeros(3,1);

    x               = zeros(1, 2*length(beta_grid));    % obst avoid Ct feature vector

    % initialize ox3 and v3:
    ox3(1:2,1)      = obs-start(1,:)';
    v3(1:2,1)       = zeros(2,1);

    for i=1:traj_length
        for j=1:length(beta_grid)
            ct3         = computeAksharaHumanoids2014ObstAvoidCtPoint( beta_grid(j,1), k_grid(j,1), ox3, v3 );
            x(1,((j-1)*2+1):((j-1)*2+2))    = ct3(1:2,1)';
        end
        ct              = x*w_Ct;
        Ct(i,:)         = ct;
        for d=1:2
            [y,yd,ydd,f] = dcp_franzi('run',d,tau,dt,ct(1,d));

            Y(i,d)      = y;
            Yd(i,d)     = yd;
            Ydd(i,d)    = ydd;
            F(i,d)      = f;
        end
        ox3(1:2,1)      = obs-Y(i,:)';
        v3(1:2,1)       = Yd(i,:)';
    end
end

