function [ w, nmse, Ct_fit ] = learnUsingLASSO( X, Ct )
    w           = zeros(size(X,2), size(Ct,2));
    alpha       = 1;
    num_lambda  = 3;
    
    for d=1:size(Ct,2)
        display(['LASSO dim: ', num2str(d)]);
        [w_lasso_d, FitInfo]    = lasso( X, Ct(:,d), 'Alpha', alpha, 'NumLambda', num_lambda );
        w(:,d)                  = w_lasso_d(:,1);   % pick the column of w_lasso_d corresponding to the smallest regularization constant
    end
    
    [ mse, nmse, Ct_fit ]       = computeNMSE( X, w, Ct );
end

