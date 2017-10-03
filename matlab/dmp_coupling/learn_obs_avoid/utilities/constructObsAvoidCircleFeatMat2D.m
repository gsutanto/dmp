function [ X, T_x3, T_v3 ] = constructObsAvoidCircleFeatMat2D( Yo, Yod, obs, beta_grid, k_grid, obs_radius )
    is_y_yd_shifted = 1;

    traj_length     = size(Yo,1);

    o3              = zeros(3,1);
    x3              = zeros(3,1);
    v3              = zeros(3,1);

    % Observed variable logging:
    T_x3            = zeros(traj_length,2);
    T_v3            = zeros(traj_length,2);
    
    Yo_shifted              = Yo;
    Yod_shifted             = Yod;
    if (is_y_yd_shifted)
        Yo_shifted(2:end,:) = Yo_shifted(1:end-1,:);
        Yod_shifted(2:end,:)= Yod_shifted(1:end-1,:);
    end

    X               = zeros(traj_length, 2*length(beta_grid));

    % First unroll the obstacle avoidance trajectory:
    for i=1:traj_length
        o3(1:2,1)       = obs;
        x3(1:2,1)       = Yo_shifted(i,:)';
        v3(1:2,1)       = Yod_shifted(i,:)';

        T_x3(i,:)       = x3(1:2,1)';
        T_v3(i,:)       = v3(1:2,1)';

        % x3 and v3 computed here is the "observed" variable (in HMM sense).

        for j=1:length(beta_grid)
            ct3         = computeAksharaHumanoids2014ObstAvoidCtSphere( beta_grid(j,1), k_grid(j,1), x3, v3, o3, obs_radius );
            X(i,((j-1)*2+1):((j-1)*2+2)) = ct3(1:2,1)';
        end
    end
end
