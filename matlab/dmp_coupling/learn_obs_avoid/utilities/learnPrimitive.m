function [ w, Yfit, Ydfit, Yddfit, Ftarget, Ffit2 ] = learnPrimitive( Yi, Ydi, Yddi, n_rfs, start, goal, traj_length, dt, c_order )
    
    global dcps;
    
    tau             = (traj_length-1)*dt;
    w               = zeros(n_rfs,2);

    Yfit            = zeros(traj_length,2);
    Ydfit           = zeros(traj_length,2);
    Yddfit          = zeros(traj_length,2);
    
    Ftarget         = zeros(traj_length,2);
    Ffit            = zeros(traj_length,2);
    Ffit2           = zeros(traj_length,2);
    
    for d=1:2
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d, start(1,d));
        dcp_franzi('set_goal',d,goal(1,d),1);
            
%         [Yfit(:,d),Ydfit(:,d),Yddfit(:,d),Ftarget(:,d),Ffit(:,d),X] = dcp_franzi('batch_fit',ID,tau,dt,Yi(:,d),Ydi(:,d),Yddi(:,d));
        [w(:,d),Ftarget(:,d),Ffit(:,d)] = dcp_franzi('batch_fit',d,tau,dt,Yi(:,d),Ydi(:,d),Yddi(:,d),start(1,d),goal(1,d));
        
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d, start(1,d));
        dcp_franzi('set_goal',d,goal(1,d),1);
        dcps(d).w   = w(:,d);
        
        for i=1:traj_length
            [y,yd,ydd,f,x] = dcp_franzi('run',d,tau,dt);

            Yfit(i,d)   = y;
            Ydfit(i,d)  = yd;
            Yddfit(i,d) = ydd;
            
            Ffit2(i,d)  = f;
        end
    end
end