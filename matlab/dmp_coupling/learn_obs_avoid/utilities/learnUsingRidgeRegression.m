function [ w, nmse, Ct_fit ] = learnUsingRidgeRegression( X, Ct )
    w   = zeros(size(X,2), size(Ct,2));
    
    XX  = X.'*X;
    xc  = X.'*Ct;

    reg = 1e-9;
    A   = reg*eye(size(XX,2));
    
    w   = (A + XX)\xc;
    
    [ mse, nmse, Ct_fit ]   = computeNMSE( X, w, Ct );
end

