function [ varargout ] = learnMultiDimensionalPrimitiveMulti( multi_dim_traj, ...
                                                              dt, n_rfs, c_order, ...
                                                              varargin )
    % Author: Giovanni Sutanto
    % Date  : February 2017
    if (nargin > 4)
        unroll_traj_length  = varargin{1};
    else
        unroll_traj_length  = -1;
    end
    if (nargin > 5)
        is_ensuring_initial_state_continuity    = varargin{2};
    else
        is_ensuring_initial_state_continuity    = 1;
    end
    
    D                   = size(multi_dim_traj{1,1},1);
    N_demo              = size(multi_dim_traj,2);
    
    taus                = zeros(1, N_demo);
    dts                 = zeros(1, N_demo);
    mean_tau            = 0.0;
    mean_start          = zeros(D,1);
    mean_goal           = zeros(D,1);
    for i=1:N_demo
        disp(['   Trajectory ', num2str(i), '/', num2str(N_demo)]);
        traj_length_i   = size(multi_dim_traj{1,i},2);
        taus(1,i)       = ((traj_length_i-1)*dt);
        dts(1,i)        = dt;
        mean_tau        = mean_tau + taus(1,i);
        mean_start      = mean_start + (multi_dim_traj{1,i}(:,1));
        mean_goal       = mean_goal + (multi_dim_traj{1,i}(:,end));
%         keyboard;
    end
    mean_tau            = mean_tau / N_demo;
    mean_start          = mean_start / N_demo;
    mean_goal           = mean_goal / N_demo;
    mean_traj_length    = round(mean_tau / dt) + 1;
    
    w                   = zeros(n_rfs,D);
        
    Ts                  = cell(1,N_demo);
    Tds                 = cell(1,N_demo);
    Tdds                = cell(1,N_demo);
    
%     fc                  = 40;                   % 40 Hz cutoff frequency
%     fs                  = 1/dt;                 % sampling frequency
%     [b, a]              = butter(2, fc/(fs/2)); % 2nd order Butterworth filter
    
    for d=1:D
        % learn primitive (per axis):
        dcp_franzi('init', d, n_rfs, num2str(d), c_order);
        
        for i=1:N_demo
            Ts{1,i}   	= multi_dim_traj{1,i}(d,:)';
%             Ts{1,i}   	= filtfilt(b, a, smoothStartEnd1DPositionProfile(multi_dim_traj{1,i}(d,:)',5));
%             Ts{1,i}   	= multi_dim_traj{1,i}(d,:)';
%             Tds{1,i}   	= filtfilt(b, a, diffnc(Ts{1,i}, dt));
            Tds{1,i}   	= diffnc(Ts{1,i}, dt);
%             Tdds{1,i}  	= filtfilt(b, a, diffnc(Tds{1,i}, dt));
            Tdds{1,i}  	= diffnc(Tds{1,i}, dt);
        end

        [w(:,d)]   = dcp_franzi('batch_fit_multi',d,taus,dts,Ts,Tds,Tdds,is_ensuring_initial_state_continuity);
    end
    
    multi_dim_dmp_params.dt                 = dt;
    multi_dim_dmp_params.n_rfs              = n_rfs;
    multi_dim_dmp_params.c_order            = c_order;
    multi_dim_dmp_params.w                  = w;
    multi_dim_dmp_params.mean_tau           = mean_tau;
    multi_dim_dmp_params.mean_start         = mean_start;
    multi_dim_dmp_params.mean_goal          = mean_goal;
    
    varargout(1)        = {multi_dim_dmp_params};
    
    if (nargout > 1)
        if (unroll_traj_length == -1)
            unroll_traj_length          = mean_traj_length;
            unroll_params.traj_length   = unroll_traj_length;
            unroll_params.tau           = mean_tau;
        else
            unroll_params.traj_length   = unroll_traj_length;
            unroll_params.tau           = dt * (unroll_traj_length - 1);
        end
        unroll_params.start             = mean_start;
        unroll_params.goal              = mean_goal;
        
        [multi_dim_dmp_unroll_fit_traj, ...
         Ffit]  = unrollNominalMultiDimensionalPrimitive( multi_dim_dmp_params, ...
                                                          unroll_params );
        
        varargout(2)    = {multi_dim_dmp_unroll_fit_traj};
        varargout(3)    = {Ffit};
    end
end

