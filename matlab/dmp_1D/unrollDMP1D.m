function [ Y, Yd, Ydd, F ] = unrollDMP1D( w, n_rfs, c_order, start, goal, dt, tau )

    global      dcps;
    traj_length = (tau/dt) + 1;
    
    Y           = zeros(traj_length, 1);
    Yd          = zeros(traj_length, 1);
    Ydd         = zeros(traj_length, 1);
    F           = zeros(traj_length, 1);

    ID          = 1;
    dcp_franzi('init', ID, n_rfs, num2str(ID), c_order);
    dcp_franzi('reset_state', ID, start);
    dcp_franzi('set_goal', ID, goal, 1);
    dcps(ID).w  = w;

    for k=1:traj_length
        [y,yd,ydd,f]    = dcp_franzi('run', ID, tau, dt);

        Y(k,1)  = y;
        Yd(k,1) = yd;
        Ydd(k,1)= ydd;

        F(k,1)  = f;
    end
    
end

