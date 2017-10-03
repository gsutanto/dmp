function [ Ct_target, F ] = computeDMPCtTarget( Yc, Ycd, Ycdd, w_baseline_DMP, n_rfs, start, goal, dt, c_order )
    D                   = size(Yc,2);

    traj_length         = size(Yc,1);
    tau                 = (traj_length-1)*dt;
    
    Ct_target           = zeros(traj_length,D);
    F                   = zeros(traj_length,D);
    
    for d=1:D
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state',d,start(d,1));
        dcp_franzi('set_goal',d,goal(d,1),1);
        
        ID              = d;
        [ Ct_target(:,d), F(:,d) ] = dcp_franzi('batch_compute_target_ct',ID,tau,dt,w_baseline_DMP(:,d),Yc(:,d),Ycd(:,d),Ycdd(:,d),goal(d,1),1);
    end
end