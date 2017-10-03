function [ varargout ] = unrollNominalMultiDimensionalPrimitive( multi_dim_dmp_params, ...
                                                                 unroll_params )
    % Author: Giovanni Sutanto
    % Date  : February 2017
    
    global                  dcps;
    
    D                       = size(multi_dim_dmp_params.w, 2);
    
    dt                      = multi_dim_dmp_params.dt;
    n_rfs                   = multi_dim_dmp_params.n_rfs;
    c_order                 = multi_dim_dmp_params.c_order;
    w                       = multi_dim_dmp_params.w;
    
    unroll_tau              = unroll_params.tau;
    unroll_traj_length      = unroll_params.traj_length;
    unroll_start            = unroll_params.start;
    unroll_goal             = unroll_params.goal;
    
    if (unroll_traj_length <= 0)
        unroll_traj_length  = round(unroll_tau / dt) + 1;
    end

    Y_fit           = zeros(unroll_traj_length, D);
    Yd_fit          = zeros(unroll_traj_length, D);
    Ydd_fit         = zeros(unroll_traj_length, D);

    Ffit          	= zeros(unroll_traj_length, D);
    X               = zeros(unroll_traj_length, 1);
    V               = zeros(unroll_traj_length, 1);
    PSI             = zeros(unroll_traj_length, n_rfs);
    time_unroll     = zeros(unroll_traj_length, 1);

    % t_unroll==0 corresponds to initial conditions
    t_unroll        = 0;
    for d=1:D
        % unrolling (per axis):
        dcp_franzi('init', d, n_rfs, num2str(d), c_order);
        dcp_franzi('reset_state', d, unroll_start(d,1));
        dcp_franzi('set_goal', d, unroll_goal(d,1), 1);
        dcps(d).w   = w(:,d);
        
        y           = dcps(d).y;
        yd          = dcps(d).yd;
        ydd         = dcps(d).ydd;
        f           = dcps(d).f;
        x           = dcps(d).x;
        v           = dcps(d).v;
        psi         = dcps(d).psi;

        for k=1:unroll_traj_length
            if (d==1)
                X(k,1)          = x;
                V(k,1)          = v;
                PSI(k,:)        = psi;
                t_unroll        = t_unroll + dt;

                time_unroll(k,1)= t_unroll;
            end

            Y_fit(k,d)          = y;
            Yd_fit(k,d)         = yd;
            Ydd_fit(k,d)        = ydd;

            Ffit(k,d)           = f;
            
            [y,yd,ydd,f,x,v,psi]= dcp_franzi('run',d,unroll_tau,dt);
        end
    end

    multi_dim_dmp_unroll_fit_traj       = cell(4,1);
    multi_dim_dmp_unroll_fit_traj{1,1}  = Y_fit;
    multi_dim_dmp_unroll_fit_traj{2,1}  = Yd_fit;
    multi_dim_dmp_unroll_fit_traj{3,1}  = Ydd_fit;
    multi_dim_dmp_unroll_fit_traj{4,1}  = time_unroll;

    varargout(1)    = {multi_dim_dmp_unroll_fit_traj};
    varargout(2)    = {Ffit};
    varargout(3)    = {X};
    varargout(4)    = {V};
    varargout(5)    = {PSI};
end