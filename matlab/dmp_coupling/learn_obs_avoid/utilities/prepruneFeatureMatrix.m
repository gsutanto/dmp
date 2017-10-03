function [ retained_feature_idx ] = prepruneFeatureMatrix( varargin )
    X                           = varargin{1};
    if (nargin > 1)
        % threshold_mode == 1: threshold on feature range (max-min)
        % threshold_mode == 2: threshold on feature variance (hard threshold on variance)
        % threshold_mode == 3: threshold on feature variance
        %                      (threshold min_feat_relative_var, such that 
        %                       max_feat_var/min_feat_relative_var ~ variance_ratio)
        % threshold_mode == 4: two thresholds:
        %                      (1) threshold on individual feature variance
        %                          i.e. any feature with variance less than
        %                          multiplier_feat_var_thres*mean_feat_var/(max_feat_var/mean_feat_var)
        %                          will be thrown out
        %                      (2) threshold on pairwise feature distance
        %                          i.e. if a pair of features has distance 
        %                          less than
        %                          multiplier_feat_dist_thres*mean_feat_dist/(max_feat_dist/mean_feat_dist),
        %                          then one of these features will be
        %                          thrown out
        % threshold_mode == 5: two thresholds:
        %                      (1) threshold on individual feature variance
        %                          (threshold min_feat_relative_var, such that 
        %                           max_feat_var/min_feat_relative_var ~ variance_ratio)
        %                      (2) threshold on pairwise feature distance
        %                          i.e. if a pair of features has distance 
        %                          less than
        %                          multiplier_feat_dist_thres*mean_feat_dist/(max_feat_dist/mean_feat_dist),
        %                          then one of these features will be
        %                          thrown out
        % threshold_mode == 6: three thresholds:
        %                      (1) & (2) Two thresholds of threshold_mode == 5
        %                      (3) VIF iterative thresholding until
        %                          condition number is below cond_thresh
        % threshold_mode == 7: three thresholds:
        %                      (1) & (2) Two thresholds of threshold_mode == 4
        %                      (3) VIF iterative thresholding until
        %                          condition number is below cond_thresh
        % threshold_mode == 8: threshold based on Stepwise Regression
        %                      result: those at regression history in which
        %                      the weights didn't explode
        threshold_mode          = varargin{2};
    else
        threshold_mode          = 1;
    end
    
    min_feat_range              = 1e-3;
    min_feat_variance           = 5e-1;
    variance_ratio              = 500.0;
    
    max_feat_var_thres          = 1.0;
    multiplier_feat_var_thres   = 1.0;
    
    max_feat_dist_thres         = 100.0;
    multiplier_feat_dist_thres  = 1.0;
    
    cond_thresh                 = 100.0;
    
    if (threshold_mode == 1)
        rind   	= find(range(X,1) > min_feat_range);
    elseif (threshold_mode == 2)
        rind   	= find(var(X,0,1) > min_feat_variance);
    elseif (threshold_mode == 3)
        max_feat_var            = max(var(X,0,1));
        min_feat_relative_var   = max_feat_var/variance_ratio;
        
        rind   	= find(var(X,0,1) >= min_feat_relative_var);
    elseif ((threshold_mode >= 4) && (threshold_mode <= 7))
        feat_variance           = var(X,0,1);
        max_feat_var            = max(feat_variance);
        mean_feat_var           = mean(feat_variance);
        if ((threshold_mode == 4) || (threshold_mode == 7))
            feat_var_thres     	= multiplier_feat_var_thres * mean_feat_var/(max_feat_var/mean_feat_var);
            feat_var_thres     	= min(feat_var_thres, max_feat_var_thres);
            rind    = find(feat_variance >= feat_var_thres);
        elseif ((threshold_mode == 5) || (threshold_mode == 6))
            min_feat_relative_var   = max_feat_var/variance_ratio;
        
            rind   	= find(feat_variance >= min_feat_relative_var);
        end
        
        X_reduced               = X(:,rind);
        pairwise_feat_distance  = pdist2(X_reduced.', X_reduced.');
        max_feat_dist           = max(max(pairwise_feat_distance));
        mean_feat_dist          = mean(mean(pairwise_feat_distance));
        feat_dist_thres         = multiplier_feat_dist_thres * mean_feat_dist/(max_feat_dist/mean_feat_dist);
        feat_dist_thres         = min(feat_dist_thres, max_feat_dist_thres);
        L                       = tril(pairwise_feat_distance);
        [i,j]                   = find(L < feat_dist_thres);
        idx_prune               = [];
        for t=1:length(i)
            if (i(t) > j(t))
                idx_prune       = [idx_prune, i(t)];
            end
        end
        idx_retain              = setdiff([1:length(rind)], idx_prune);
        rind    = rind(idx_retain);
        
        if ((threshold_mode == 6) || (threshold_mode == 7))
            X_reduced        	= X(:,rind);
            vif_iter            = 1;
            cond_X_reduced      = cond(X_reduced);
            while (cond_X_reduced > cond_thresh)
                v   = vif(X_reduced);
                [max_vif, max_vif_idx]  = max(v);
                disp(['vif iter=',num2str(vif_iter),...
                      ', cond(X_reduced)=',num2str(cond_X_reduced),...
                      ', max(vif)=',num2str(max_vif)]);
                vif_prune_idx   = rind(max_vif_idx);
                rind            = setdiff(rind, vif_prune_idx);
                X_reduced       = X(:,rind);
                vif_iter        = vif_iter + 1;
                cond_X_reduced  = cond(X_reduced);
            end
        end
    elseif (threshold_mode == 8)
        load('4_rind_based_on_stepwisefit_result.mat');
    end
    
    retained_feature_idx        = rind;
end