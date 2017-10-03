function [ X_dim_reduced, mu_X, pca_projection_matrix ] = performDimReductionWithPCA( X, percent_variance_explained_thresh )
    [coeff,score,latent,tsquared,explained,mu_X] = pca(X);
    N_data              = size(X,1);
    N_orig_feature_dim  = size(X,2);
    for ndim=1:N_orig_feature_dim
        if(sum(explained(1:ndim,1)) > percent_variance_explained_thresh)
            N_reduced_dim   = ndim;
            break; 
        end;
    end
    pca_projection_matrix   = coeff(:,(1:N_reduced_dim));
    X_dim_reduced           = (X-repmat(mu_X,N_data,1))*pca_projection_matrix;
end