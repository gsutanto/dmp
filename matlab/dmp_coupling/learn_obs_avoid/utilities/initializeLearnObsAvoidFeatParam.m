function [ loa_feat_param ] = initializeLearnObsAvoidFeatParam( varargin )
    loa_feat_param.N_loa_feat_cols_per_point_axis   = 0;
    if (nargin > 0)
        loa_feat_param.c_order                      = varargin{1};
        if (loa_feat_param.c_order == 1)
            loa_feat_param.alpha_v 	= 25.0;
            loa_feat_param.beta_v 	= loa_feat_param.alpha_v/4.0;
        else
            loa_feat_param.alpha_x  = 25.0/3.0;
        end
    end
    if (nargin > 1)
        loa_feat_param.learning_constraint_mode     = varargin{2};
    end
    if (nargin > 2)
        loa_feat_param.is_tau_invariant             = varargin{3};
    else
        loa_feat_param.is_tau_invariant             = 1;    % default is tau-invariant
    end
end