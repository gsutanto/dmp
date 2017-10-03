function [ w, Yifit, Ydifit, Yddifit, Fifit, mean_start, mean_goal, mean_dt ] = learnPrimitiveMulti( Yis, Ydis, Yddis, taus, dts, n_rfs, c_order )
    
    global          dcps;
    
    D               = size(Yis{1,1},2);
    
    w               = zeros(n_rfs,D);

    mean_tau        = mean(taus);
    mean_dt         = mean(dts);
    mean_traj_length= round(mean_tau/mean_dt);
    mean_start      = zeros(D,1);
    mean_goal       = zeros(D,1);

    Yifit           = zeros(mean_traj_length,D);
    Ydifit          = zeros(mean_traj_length,D);
    Yddifit         = zeros(mean_traj_length,D);
    
    Fifit           = zeros(mean_traj_length,D);
    
    for d=1:D
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        
        for j=1:size(Yis,2)
            Ts{1,j}         = Yis{1,j}(:,d);
            Tds{1,j}        = Ydis{1,j}(:,d);
            Tdds{1,j}       = Yddis{1,j}(:,d);
            mean_start(d,1) = mean_start(d,1) + Yis{1,j}(1,d);
            mean_goal(d,1)  = mean_goal(d,1) + Yis{1,j}(end,d);
        end

        [w(:,d)]            = dcp_franzi('batch_fit_multi',d,taus,dts,Ts,Tds,Tdds);

        mean_start(d,1)     = mean_start(d,1)/size(Yis,2);
        mean_goal(d,1)      = mean_goal(d,1)/size(Yis,2);
        
        dcp_franzi('init',d,n_rfs,num2str(d), c_order);
        dcp_franzi('reset_state', d, mean_start(d,1));
        dcp_franzi('set_goal', d, mean_goal(d,1), 1);
        dcps(d).w   = w(:,d);
        
        for k=1:mean_traj_length
            [y,yd,ydd,f]    = dcp_franzi('run',d,mean_tau,mean_dt);

            Yifit(k,d)      = y;
            Ydifit(k,d)     = yd;
            Yddifit(k,d)    = ydd;
            
            Fifit(k,d)      = f;
        end
    end
end