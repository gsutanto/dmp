function [ pc_projector, X_mean_row_vector ] = performPCA( X, max_cond_number )
    N               	= size(X,1);
    
    % zero-mean the feature matrix:
    X_mean_row_vector 	= mean(X,1);
    X_zero_mean       	= X - repmat(X_mean_row_vector,N,1);
    
    % perform Singular Value Decomposition (SVD):
    [U,S,V]           	= svd(X_zero_mean,'econ');
    s                   = diag(S);
    
    for i=2:length(s)
        if ((s(1,1)/s(i,1)) > max_cond_number)
            cutoff_idx  = i-1;
            break;
        elseif (i == length(s))
            cutoff_idx  = length(s);    % use all feature dimensions (no dimensionality reduction)
        end
    end
    
    pc_projector        = V(:,1:cutoff_idx);
end