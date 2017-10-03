function [ loa_feat_param ] = initializePF_DYN4LearnObsAvoidFeatParam( varargin )
    loa_feat_param                          = varargin{1};
    N_PF_DYN4_k_grid                        = varargin{2};
    PF_DYN4_k_low                           = varargin{3};
    PF_DYN4_k_high                          = varargin{4};
    N_PF_DYN4_s_grid                        = varargin{5};
    loa_feat_param.c_order                  = varargin{6};
    if (nargin > 6)
        loa_feat_param.feat_constraint_mode = varargin{7};
    else
        loa_feat_param.feat_constraint_mode = '_UNCONSTRAINED_';
    end
    if (nargin > 7)
        PF_DYN4_s_low                     	= varargin{8};
        PF_DYN4_s_high                     	= varargin{9};
    else
        PF_DYN4_s_low                     	= 0.0;
        PF_DYN4_s_high                     	= 1.0;
    end
    if (nargin > 9)
        PF_DYN4_s_default_D                 = varargin{10};
    else
        PF_DYN4_s_default_D                 = 1.0;
    end
    
    loa_feat_param.PF_DYN4_N_k_grid         = N_PF_DYN4_k_grid;
    loa_feat_param.PF_DYN4_N_s_grid         = N_PF_DYN4_s_grid;
    
    PF_DYN4_k_row_grid                      = (1.0./((linspace(PF_DYN4_k_low, PF_DYN4_k_high, N_PF_DYN4_k_grid)).^2));
    PF_DYN4_k_row_col_grid                  = repmat(PF_DYN4_k_row_grid, N_PF_DYN4_s_grid, 1);
    
    if (N_PF_DYN4_s_grid > 1)
        t   = (PF_DYN4_s_low:(PF_DYN4_s_high-PF_DYN4_s_low)/(N_PF_DYN4_s_grid-1):PF_DYN4_s_high)*0.5;
    else
        t   = PF_DYN4_s_high*0.5;
    end
    
    if (loa_feat_param.c_order == 1)
        loa_feat_param.alpha_v 	= 25.0;
        loa_feat_param.beta_v 	= loa_feat_param.alpha_v/4.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = 1-(1+alpha/2*t)*exp(-alpha/2*t)
        loa_feat_param.PF_DYN4_s_col_grid   = (1+((loa_feat_param.alpha_v/2)*t)).*exp(-(loa_feat_param.alpha_v/2)*t);
    else
        loa_feat_param.alpha_x  = 25.0/3.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = exp(-alpha*t)
        loa_feat_param.PF_DYN4_s_col_grid   = exp(-loa_feat_param.alpha_x*t);
    end
    if (N_PF_DYN4_s_grid > 1)
        loa_feat_param.PF_DYN4_s_col_D_grid = (diff(loa_feat_param.PF_DYN4_s_col_grid)*0.55).^2;
        loa_feat_param.PF_DYN4_s_col_D_grid = 1./[loa_feat_param.PF_DYN4_s_col_D_grid, loa_feat_param.PF_DYN4_s_col_D_grid(end)];
    else
        loa_feat_param.PF_DYN4_s_col_D_grid = PF_DYN4_s_default_D;
    end
    
    PF_DYN4_s_row_col_grid      = repmat(loa_feat_param.PF_DYN4_s_col_grid, 1, N_PF_DYN4_k_grid);
    PF_DYN4_s_row_col_D_grid    = repmat(loa_feat_param.PF_DYN4_s_col_D_grid, 1, N_PF_DYN4_k_grid);
    
    loa_feat_param.PF_DYN4_k_rowcol_vector      = reshape(PF_DYN4_k_row_col_grid, N_PF_DYN4_s_grid*N_PF_DYN4_k_grid, 1);
    loa_feat_param.PF_DYN4_s_rowcol_vector      = reshape(PF_DYN4_s_row_col_grid, N_PF_DYN4_s_grid*N_PF_DYN4_k_grid, 1);
    loa_feat_param.PF_DYN4_s_rowcol_D_vector    = reshape(PF_DYN4_s_row_col_D_grid, N_PF_DYN4_s_grid*N_PF_DYN4_k_grid, 1);

    loa_feat_param.PF_DYN4_N_loa_feat_cols_per_point_axis   = (N_PF_DYN4_s_grid * N_PF_DYN4_k_grid);
end