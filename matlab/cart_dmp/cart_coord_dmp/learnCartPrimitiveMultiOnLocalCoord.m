function [ varargout ] = learnCartPrimitiveMultiOnLocalCoord( varargin )
    % Author: Giovanni Sutanto
    % Date  : August 02, 2016
    
    cart_global_traj= varargin{1};
    train_data_dt  	= varargin{2};
    n_rfs           = varargin{3};
    c_order         = varargin{4};
    if (nargin > 4)
        ctraj_local_coordinate_frame_selection  = varargin{5};
    else
        ctraj_local_coordinate_frame_selection  = 1;    % default is gsutanto's ctraj_local_coordinate_frame_selection
    end
    if (nargin > 5)
        unroll_traj_length  = varargin{6};
    else
        unroll_traj_length  = -1;
    end
    if (nargin > 6)
        unroll_dt           = varargin{7};
    else
        unroll_dt           = train_data_dt;
    end
    if (nargin > 7)
        is_using_scaling    = varargin{8};
    else
        is_using_scaling    = zeros(1,3);   % default is NOT using scaling on DMPs
    end
    
    D               = 3;
    N_demo          = size(cart_global_traj,2);
    
    cart_local_traj     = cell(size(cart_global_traj));
    T_local_to_global_H = cell(1, N_demo);
    T_global_to_local_H = cell(1, N_demo);
    taus                = zeros(1, N_demo);
    dts                 = zeros(1, N_demo);
    mean_tau            = 0.0;
    mean_start_global   = zeros(3,1);
    mean_goal_global    = zeros(3,1);
    for i=1:N_demo
%         disp(['   Trajectory ', num2str(i), '/', num2str(N_demo)]);
        [cart_local_traj(1:3,i), T_local_to_global_H{1,i}, T_global_to_local_H{1,i}] = computeCartLocalTraj(cart_global_traj(1:3,i), 0, ctraj_local_coordinate_frame_selection);
        traj_length_i       = size(cart_local_traj{1,i},1);
        taus(1,i)           = ((traj_length_i-1)*train_data_dt);
        dts(1,i)            = train_data_dt;
        mean_tau            = mean_tau + taus(1,i);
        mean_start_global   = mean_start_global + (cart_global_traj{1,i}(1,:).');
        mean_goal_global    = mean_goal_global + (cart_global_traj{1,i}(end,:).');
%         keyboard;
    end
    mean_tau            = mean_tau / N_demo;
    mean_start_global   = mean_start_global / N_demo;
    mean_goal_global    = mean_goal_global / N_demo;
    
    w                   = zeros(n_rfs,D);
    dG                  = zeros(1,D);
        
    Ts                  = cell(1,N_demo);
    Tds                 = cell(1,N_demo);
    Tdds                = cell(1,N_demo);
    
    for d=1:D
        % learn primitive (per axis):
        dcp_franzi('init', d, n_rfs, num2str(d), c_order);
        
        for i=1:N_demo
            Ts{1,i}   	= cart_local_traj{1,i}(:,d);
            Tds{1,i}   	= cart_local_traj{2,i}(:,d);
            Tdds{1,i}  	= cart_local_traj{3,i}(:,d);
        end

        [w(:,d),~,~,~,~,~,~,~,~,...
         dG(:,d)]      	= dcp_franzi('batch_fit_multi',d,taus,dts,Ts,Tds,Tdds);
        
        if (is_using_scaling(1,d) == 0) % if NOT using scaling on this dimension's DMP:
            dG(:,d)     = 0;
        end
    end
    
%     cart_coord_dmp_params_basic.dt                  = train_data_dt;
    cart_coord_dmp_params_basic.dt                  = unroll_dt;
    cart_coord_dmp_params_basic.n_rfs               = n_rfs;
    cart_coord_dmp_params_basic.c_order             = c_order;
    cart_coord_dmp_params_basic.w                   = w;
    cart_coord_dmp_params_basic.dG                  = dG;
    unroll_cart_coord_params_basic.mean_tau         = mean_tau;
    unroll_cart_coord_params_basic.mean_start_global= mean_start_global;
    unroll_cart_coord_params_basic.mean_goal_global = mean_goal_global;
    unroll_cart_coord_params_basic.ctraj_local_coordinate_frame_selection   = ctraj_local_coordinate_frame_selection;
    
    if (nargout > 1)
        if (unroll_traj_length == -1)
            unroll_cart_coord_params_basic.mean_tau = mean_tau;
        else
            unroll_cart_coord_params_basic.mean_tau = unroll_dt * (unroll_traj_length - 1);
        end
        
        [cart_coord_dmp_params, ...
         cart_coord_dmp_unroll_fit_global_traj, ...
         cart_coord_dmp_unroll_fit_local_traj, ...
         Ffit]  = unrollCartPrimitiveOnLocalCoord( cart_coord_dmp_params_basic, ...
                                                   unroll_cart_coord_params_basic );
        
        varargout(1)    = {cart_coord_dmp_params};
        varargout(2)    = {cart_coord_dmp_unroll_fit_global_traj};
        varargout(3)    = {cart_coord_dmp_unroll_fit_local_traj};
        varargout(4)    = {Ffit};
        varargout(5)    = {mean_tau};
    else
        cart_coord_dmp_params   = completeCartCoordDMPParams( cart_coord_dmp_params_basic,...
                                                              unroll_cart_coord_params_basic );

        varargout(1)        	= {cart_coord_dmp_params};
    end
end