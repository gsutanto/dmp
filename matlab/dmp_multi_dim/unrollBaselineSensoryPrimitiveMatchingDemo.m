function [ varargout ] = unrollBaselineSensoryPrimitiveMatchingDemo( sensory_dmp_params, ...
                                                                     demo_sensory_trace, ...
                                                                     varargin )
    if (nargin > 2)
        is_plotting_unroll_and_demo_comparison  = varargin{1};
    else
        is_plotting_unroll_and_demo_comparison  = 0;
    end
    sensory_unroll_params.traj_length   = size(demo_sensory_trace, 1);
    sensory_unroll_params.tau           = sensory_dmp_params.dt * (sensory_unroll_params.traj_length - 1);
    sensory_unroll_params.start         = sensory_dmp_params.mean_start;
    sensory_unroll_params.goal          = sensory_dmp_params.mean_goal;
    [sensory_dmp_unroll_fit_traj, ...
     ~, ...
     X, ...
     V, ...
     PSI]   = unrollNominalMultiDimensionalPrimitive( sensory_dmp_params, ...
                                                      sensory_unroll_params );
    unrolled_nominal_sensory_trace      = sensory_dmp_unroll_fit_traj{1,1};
    
    if (is_plotting_unroll_and_demo_comparison == 3)
        fc      = 3;
        fs      = 1/sensory_dmp_params.dt;
        [b, a] 	= butter(2, fc/(fs/2), 'high');
    end
    
    if (is_plotting_unroll_and_demo_comparison)
        figure;
        D   = size(demo_sensory_trace,2);
        N_plot_cols = ceil(D/5);
        for d=1:D
            subplot(ceil(D/N_plot_cols),N_plot_cols,d);
            hold on;
                if (is_plotting_unroll_and_demo_comparison == 1)
                    pd  = plot(demo_sensory_trace(:,d));
                    pu  = plot(unrolled_nominal_sensory_trace(:,d));
                elseif (is_plotting_unroll_and_demo_comparison == 2)    % initial offset cancellation
                    pd  = plot(demo_sensory_trace(:,d)-mean(demo_sensory_trace(1:50,d)));
                    pu  = plot(unrolled_nominal_sensory_trace(:,d)-mean(unrolled_nominal_sensory_trace(1:50,d)));
                elseif (is_plotting_unroll_and_demo_comparison == 3)    % high-pass filtering
                    pd  = plot(filtfilt(b, a, demo_sensory_trace(:,d)));
                    pu  = plot(filtfilt(b, a, unrolled_nominal_sensory_trace(:,d)));
                end
                if (d == 1)
                    legend([pd,pu], 'corrected\_sensory\_demo', 'nominal\_sensory\_unroll');
                end
            hold off;
        end
        keyboard;
    end
    
    varargout(1)    = {unrolled_nominal_sensory_trace};
    varargout(2)    = {X};
    varargout(3)    = {V};
    varargout(4)    = {PSI};
end

