function [ w, nmse, Ct_fit, mse ] = learnUsingARD( varargin )
    X                           = varargin{1};
    Ct                          = varargin{2};
    if (nargin > 2)
        max_abs_weight_threshold= varargin{3};
    else
    %     max_abs_weight_threshold= 1e9; % worse unrolling
        max_abs_weight_threshold= 7.5e3; % best so far
    end
    if (nargin > 3)
        N_iter                  = varargin{4};
    else
        N_iter                  = 200;
    end
    if (nargin > 4)
        rind                    = varargin{5};
    else
        rind                    = prepruneFeatureMatrix(X, 1);
    end
    
    D                           = size(Ct,2);
    
    w                           = zeros(size(X,2), D);
    
%     precision_cap               = 1e-8; % good value for AF_H14 model
    precision_cap               = 0;

    for d=1:D
        display(['ARD dim: ', num2str(d)]);
        [w_ard_d, r_ard_idx, log10_b_traj, log10_a_traj] = ARD( X, Ct(:,d), 0, precision_cap, max_abs_weight_threshold, N_iter, rind );
        w(r_ard_idx,d)          = w_ard_d;
    end
    
    [ mse, nmse, Ct_fit ]       = computeNMSE( X, w, Ct );
    
%     figure;
%     plot(log10_b_traj);
%     title('log10(beta) trajectory');
%     drawnow;
%     
%     figure;
%     h = surf(log10_a_traj);
%     set(h,'LineStyle','none');
%     title('log10(alpha) trajectory');
%     drawnow;
end