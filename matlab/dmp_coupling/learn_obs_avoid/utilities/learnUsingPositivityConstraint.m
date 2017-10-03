function [ w, nmse, Ct_fit, learning_param ] = learnUsingPositivityConstraint( varargin )
    X                           = varargin{1};
    Ct                          = varargin{2};
    learning_param              = varargin{3};
    
    learning_param.retain_idx   = find(var(X,0,1) > learning_param.feature_variance_threshold); % only retain weights corresponding to rich enough features (to avoid numerical instability)
    X_new                       = X(:, learning_param.retain_idx);
    X                           = X_new;
    
    D                           = size(Ct,2);
    
    w                           = zeros(size(X,2), D);

    for d=1:D
        display(['Learn w/ Positivity Constraint, dim: ', num2str(d)]);
        [w_d]   = PositivityConstrainedLearning( X, Ct(:,d) );
        w(:,d)  = w_d;
    end
    
    [ mse, nmse, Ct_fit ]       = computeNMSE( X, w, Ct );
end