function [ w, nmse, Ct_fit ] = learnUsingBayesianRegression( varargin )
    X                           = varargin{1};
    Ct                          = varargin{2};
    if (nargin > 2)
        max_abs_weight_threshold= varargin{3};
    else
    %     max_abs_weight_threshold= 1e9; % worse unrolling
        max_abs_weight_threshold= 5e3; % best so far
    end
    if (nargin > 3)
        num_iter                = varargin{4};
    else
        num_iter                = 200;
    end

    D                           = size(Ct,2);
    w                           = zeros(size(X,2), D);
    debug_interval              = 1;
    debug_mode                  = 1;
    alpha_min_threshold         = 0;
    
    for d=1:D
        display(['Bayesian Regression dim: ', num2str(d)]);
        [w_br_d, r_br_idx, cfit_hist, w_hist, log10_a_hist] = BayesianRegression( X, Ct(:,d), num_iter, debug_interval, debug_mode, alpha_min_threshold, max_abs_weight_threshold );
        w(r_br_idx,d)                                       = w_br_d;
    end
    
    [ mse, nmse, Ct_fit ]   = computeNMSE( X, w, Ct );
    
    figure;
    hold            on;
    plot(log10_a_hist');
    title('log10(a) Evolution');
    hold            off;
end

