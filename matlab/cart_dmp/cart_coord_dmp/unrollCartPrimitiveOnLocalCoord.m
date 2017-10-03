function [ varargout ] = unrollCartPrimitiveOnLocalCoord( cart_coord_dmp_params_basic, ...
                                                          unroll_cart_coord_params_basic, ...
                                                          varargin )
    % Author: Giovanni Sutanto
    % Date  : February 2017
    
    if (nargin > 2)
        Ct                  = varargin{1};
        is_using_ct         = 1;
    else
        is_using_ct         = 0;
    end
    
    cart_coord_dmp_params   = completeCartCoordDMPParams( cart_coord_dmp_params_basic,...
                                                          unroll_cart_coord_params_basic );
    
    global                  dcps;
    
    D                       = size(cart_coord_dmp_params.w, 2);
    assert(D == 3,'Weights for Cartesian Coordinate DMP could only be 3 dimensional.');
    
    dt                      = cart_coord_dmp_params.dt;
    n_rfs                   = cart_coord_dmp_params.n_rfs;
    c_order                 = cart_coord_dmp_params.c_order;
    w                       = cart_coord_dmp_params.w;
    dG                      = cart_coord_dmp_params.dG;
    
    unroll_tau              = cart_coord_dmp_params.mean_tau;
    unroll_traj_length      = round(unroll_tau / dt) + 1;
    unroll_start_local      = cart_coord_dmp_params.mean_start_local;
    unroll_goal_local       = cart_coord_dmp_params.mean_goal_local;
    if (isfield(cart_coord_dmp_params, 'yd0_local'))
        unroll_yd0_local   	= cart_coord_dmp_params.yd0_local;
    else
        unroll_yd0_local    = zeros(3,1);
    end
    if (isfield(cart_coord_dmp_params, 'ydd0_local'))
        unroll_ydd0_local   = cart_coord_dmp_params.ydd0_local;
    else
        unroll_ydd0_local   = zeros(3,1);
    end
    
    if (is_using_ct)
        assert(size(Ct, 1) == unroll_traj_length, 'Given Ct is mis-match in length!');
        assert(size(Ct, 2) == D, 'Ct for CartCoordDMP must be 3-dimensional!');
    end
    
    Y_fit_local     = zeros(unroll_traj_length, D);
    Yd_fit_local    = zeros(unroll_traj_length, D);
    Ydd_fit_local   = zeros(unroll_traj_length, D);

    Ffit          	= zeros(unroll_traj_length, D);
    time_unroll     = zeros(unroll_traj_length, 1);

    % t_unroll==0 corresponds to initial conditions
    t_unroll        = 0;
    for d=1:D
        % unrolling (per axis):
        dcp_franzi('init', d, n_rfs, num2str(d), c_order);
        dcp_franzi('reset_state', d, unroll_start_local(d,1), unroll_yd0_local(d,1), unroll_ydd0_local(d,1), unroll_tau);
        dcps(d).dG  = dG(:,d);
        dcp_franzi('set_goal', d, unroll_goal_local(d,1), 1);
        dcps(d).w   = w(:,d);

        for k=1:unroll_traj_length
            if (is_using_ct)
                ct                  = Ct(k,d);
            else
                ct                  = 0;
            end
            [y,yd,ydd,f]            = dcp_franzi('run',d,unroll_tau,dt,ct);

            if (d==1)
                t_unroll            = t_unroll + dt;

                time_unroll(k,1)    = t_unroll;
            end

            Y_fit_local(k,d)  	= y;
            Yd_fit_local(k,d)  	= yd;
            Ydd_fit_local(k,d) 	= ydd;

            Ffit(k,d)           = f;
        end
    end

    cart_coord_dmp_unroll_fit_local_traj        = cell(4,1);
    cart_coord_dmp_unroll_fit_local_traj{1,1}   = Y_fit_local;
    cart_coord_dmp_unroll_fit_local_traj{2,1}   = Yd_fit_local;
    cart_coord_dmp_unroll_fit_local_traj{3,1}   = Ydd_fit_local;
    cart_coord_dmp_unroll_fit_local_traj{4,1}   = time_unroll;

    [ cart_coord_dmp_unroll_fit_global_traj ] = convertCTrajAtOldToNewCoordSys( cart_coord_dmp_unroll_fit_local_traj, ...
                                                                                cart_coord_dmp_params.T_local_to_global_H );

    cart_coord_dmp_unroll_fit_global_traj{4,1}  = time_unroll;

    varargout(1)    = {cart_coord_dmp_params};
    varargout(2)    = {cart_coord_dmp_unroll_fit_global_traj};
    varargout(3)    = {cart_coord_dmp_unroll_fit_local_traj};
    varargout(4)    = {Ffit};
end