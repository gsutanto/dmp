function [ loa_feat_param ] = initializeFM_KGFv01LearnObsAvoidFeatParam( varargin )
    loa_feat_param                      = varargin{1};
    N_FM_KGFv01_k_grid                  = varargin{2};
    FM_KGFv01_k_low                     = varargin{3};
    FM_KGFv01_k_high                    = varargin{4};
    N_FM_KGFv01_s_grid                  = varargin{5};
    loa_feat_param.c_order              = varargin{6};

    % the only supported mode here:
    loa_feat_param.feat_constraint_mode = '_UNCONSTRAINED_';
    
    loa_feat_param.FM_KGFv01_N_k_grid   = N_FM_KGFv01_k_grid;
    loa_feat_param.FM_KGFv01_N_s_grid   = N_FM_KGFv01_s_grid;
    
    FM_KGFv01_k_row_grid                = (1.0./((linspace(FM_KGFv01_k_low, FM_KGFv01_k_high, N_FM_KGFv01_k_grid)).^2));
    FM_KGFv01_k_row_col_grid            = repmat(FM_KGFv01_k_row_grid, N_FM_KGFv01_s_grid, 1);
    
    t   = (0:1/(N_FM_KGFv01_s_grid-1):1).'*0.5;
    
    if (loa_feat_param.c_order == 1)
        loa_feat_param.alpha_v 	= 25.0;
        loa_feat_param.beta_v 	= loa_feat_param.alpha_v/4.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = 1-(1+alpha/2*t)*exp(-alpha/2*t)
        loa_feat_param.FM_KGFv01_s_col_grid = (1+((loa_feat_param.alpha_v/2)*t)).*exp(-(loa_feat_param.alpha_v/2)*t);
    else
        loa_feat_param.alpha_x  = 25.0/3.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = exp(-alpha*t)
        loa_feat_param.FM_KGFv01_s_col_grid = exp(-loa_feat_param.alpha_x*t);
    end
    loa_feat_param.FM_KGFv01_s_col_D_grid   = (diff(loa_feat_param.FM_KGFv01_s_col_grid)*0.55).^2;
    loa_feat_param.FM_KGFv01_s_col_D_grid   = 1./[loa_feat_param.FM_KGFv01_s_col_D_grid; loa_feat_param.FM_KGFv01_s_col_D_grid(end)];
    
    FM_KGFv01_s_row_col_grid    = repmat(loa_feat_param.FM_KGFv01_s_col_grid, 1, N_FM_KGFv01_k_grid);
    FM_KGFv01_s_row_col_D_grid  = repmat(loa_feat_param.FM_KGFv01_s_col_D_grid, 1, N_FM_KGFv01_k_grid);
    
    loa_feat_param.FM_KGFv01_k_rowcol_vector    = reshape(FM_KGFv01_k_row_col_grid, N_FM_KGFv01_s_grid*N_FM_KGFv01_k_grid, 1);
    loa_feat_param.FM_KGFv01_s_rowcol_vector    = reshape(FM_KGFv01_s_row_col_grid, N_FM_KGFv01_s_grid*N_FM_KGFv01_k_grid, 1);
    loa_feat_param.FM_KGFv01_s_rowcol_D_vector  = reshape(FM_KGFv01_s_row_col_D_grid, N_FM_KGFv01_s_grid*N_FM_KGFv01_k_grid, 1);

    loa_feat_param.FM_KGFv01_N_loa_feat_cols_per_point_axis   = (N_FM_KGFv01_s_grid * N_FM_KGFv01_k_grid);
end