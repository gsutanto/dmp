function [ w, nmse, Ct_fit, mse ] = learnUsingARDperAxis( varargin )
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
    
    D                           = 3;
    
    w                           = zeros(size(X,2), D);
    
%     precision_cap               = 1e-8; % good value for AF_H14 model
    precision_cap               = 0;
    
    mse                         = zeros(1, D);
    nmse                        = zeros(1, D);
    Ct_fit                      = zeros(size(Ct));

    for d=1:D
        display(['ARD dim: ', num2str(d)]);
        data_idx                = [d:D:size(X,1)];
        X_dim_d                 = X(data_idx,:);
        Ct_dim_d                = Ct(data_idx,:);
        [w_ard_d, r_ard_idx]    = ARD( X_dim_d, Ct_dim_d, 0, precision_cap, max_abs_weight_threshold, N_iter, rind );
        w(r_ard_idx,d)          = w_ard_d;
    
        [mse(1,d), nmse(1,d), Ct_fit(data_idx,:)] = computeNMSE( X_dim_d, w(:,d), Ct_dim_d );
    end
end