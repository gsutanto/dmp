function [ w, nmse, Ct_fit ] = learnUsingVBLS( X, Ct )
    w                       = zeros(size(X,2), size(Ct,2));
    result                  = cell(size(Ct,2),1);

    options.noise           = 1;        % initial output noise variance
    options.threshold       = 1e-5;     % threshold for convergence
    options.numIterations   = 10000;    % max number of EM iterations
    
    for d=1:size(Ct,2)
        result{d,1}         = vbls(X, Ct(:,d), options);
        w(:,d)              = result{d,1}.b_mean;
    end
    
    [ mse, nmse, Ct_fit ] = computeNMSE( X, w, Ct );
end

