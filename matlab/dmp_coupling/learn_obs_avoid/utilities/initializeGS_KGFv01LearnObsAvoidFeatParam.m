function [ loa_feat_param ] = initializeGS_KGFv01LearnObsAvoidFeatParam( varargin )
    loa_feat_param                          = varargin{1};
    N_GS_KGFv01_beta_grid                   = varargin{2};
    GS_KGFv01_beta_low                      = varargin{3};
    GS_KGFv01_beta_high                     = varargin{4};
    N_GS_KGFv01_k_grid                      = varargin{5};
    GS_KGFv01_k_low                         = varargin{6};
    GS_KGFv01_k_high                        = varargin{7};
    N_GS_KGFv01_s_grid                      = varargin{8};
    loa_feat_param.c_order                  = varargin{9};
    if (nargin > 9)
        loa_feat_param.feat_constraint_mode = varargin{10};
    else
        loa_feat_param.feat_constraint_mode = '_UNCONSTRAINED_';
    end
    
    loa_feat_param.GS_KGFv01_N_beta_grid    = N_GS_KGFv01_beta_grid;
    loa_feat_param.GS_KGFv01_N_k_grid       = N_GS_KGFv01_k_grid;
    loa_feat_param.GS_KGFv01_N_s_grid       = N_GS_KGFv01_s_grid;
    
    loa_feat_param.GS_KGFv01_beta_col_grid  = linspace(GS_KGFv01_beta_low, GS_KGFv01_beta_high, N_GS_KGFv01_beta_grid);
    GS_KGFv01_k_row_grid                    = (1.0./((linspace(GS_KGFv01_k_low, GS_KGFv01_k_high, N_GS_KGFv01_k_grid)).^2));
    [GS_KGFv01_beta_row_col_grid, GS_KGFv01_k_row_col_grid] = meshgrid(loa_feat_param.GS_KGFv01_beta_col_grid,...
                                                                       GS_KGFv01_k_row_grid);
    loa_feat_param.GS_KGFv01_beta_col_D_grid    = (diff(loa_feat_param.GS_KGFv01_beta_col_grid)*0.55).^2;
    loa_feat_param.GS_KGFv01_beta_col_D_grid   	= 1./[loa_feat_param.GS_KGFv01_beta_col_D_grid, loa_feat_param.GS_KGFv01_beta_col_D_grid(end)];
    GS_KGFv01_beta_row_col_D_grid 	= repmat(loa_feat_param.GS_KGFv01_beta_col_D_grid, N_GS_KGFv01_k_grid, 1);
    
    GS_KGFv01_beta_rowcol_vector 	= reshape(GS_KGFv01_beta_row_col_grid, N_GS_KGFv01_beta_grid*N_GS_KGFv01_k_grid, 1);
    GS_KGFv01_beta_rowcol_D_vector 	= reshape(GS_KGFv01_beta_row_col_D_grid, N_GS_KGFv01_beta_grid*N_GS_KGFv01_k_grid, 1);
    GS_KGFv01_k_rowcol_vector     	= reshape(GS_KGFv01_k_row_col_grid, N_GS_KGFv01_beta_grid*N_GS_KGFv01_k_grid, 1);

    t           = (0:1/(N_GS_KGFv01_s_grid-1):1)*0.5;
    
    if (loa_feat_param.c_order == 1)
        loa_feat_param.alpha_v 	= 25.0;
        loa_feat_param.beta_v 	= loa_feat_param.alpha_v/4.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = 1-(1+alpha/2*t)*exp(-alpha/2*t)
        loa_feat_param.GS_KGFv01_s_depth_grid   = (1+((loa_feat_param.alpha_v/2)*t)).*exp(-(loa_feat_param.alpha_v/2)*t);
    else
        loa_feat_param.alpha_x  = 25.0/3.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = exp(-alpha*t)
        loa_feat_param.GS_KGFv01_s_depth_grid   = exp(-loa_feat_param.alpha_x*t);
    end
    loa_feat_param.GS_KGFv01_s_depth_D_grid     = (diff(loa_feat_param.GS_KGFv01_s_depth_grid)*0.55).^2;
    loa_feat_param.GS_KGFv01_s_depth_D_grid     = 1./[loa_feat_param.GS_KGFv01_s_depth_D_grid, loa_feat_param.GS_KGFv01_s_depth_D_grid(end)];
    
    [GS_KGFv01_s_rowcol_depth_grid, GS_KGFv01_beta_rowcol_depth_grid] = meshgrid(loa_feat_param.GS_KGFv01_s_depth_grid, GS_KGFv01_beta_rowcol_vector.');
    GS_KGFv01_k_rowcol_depth_grid       = repmat(GS_KGFv01_k_rowcol_vector, 1, N_GS_KGFv01_s_grid);
    GS_KGFv01_beta_rowcol_depth_D_grid  = repmat(GS_KGFv01_beta_rowcol_D_vector, 1, N_GS_KGFv01_s_grid);
    GS_KGFv01_s_rowcol_depth_D_grid     = repmat(loa_feat_param.GS_KGFv01_s_depth_D_grid, size(GS_KGFv01_beta_rowcol_vector, 1), 1);
    
    loa_feat_param.GS_KGFv01_beta_rowcoldepth_vector    = reshape(GS_KGFv01_beta_rowcol_depth_grid, size(GS_KGFv01_beta_rowcol_depth_grid, 1)*size(GS_KGFv01_beta_rowcol_depth_grid, 2), 1);
    loa_feat_param.GS_KGFv01_beta_rowcoldepth_D_vector  = reshape(GS_KGFv01_beta_rowcol_depth_D_grid, size(GS_KGFv01_beta_rowcol_depth_D_grid, 1)*size(GS_KGFv01_beta_rowcol_depth_D_grid, 2), 1);
    loa_feat_param.GS_KGFv01_k_rowcoldepth_vector       = reshape(GS_KGFv01_k_rowcol_depth_grid, size(GS_KGFv01_k_rowcol_depth_grid, 1)*size(GS_KGFv01_k_rowcol_depth_grid, 2), 1);
    loa_feat_param.GS_KGFv01_s_rowcoldepth_vector       = reshape(GS_KGFv01_s_rowcol_depth_grid, size(GS_KGFv01_s_rowcol_depth_grid, 1)*size(GS_KGFv01_s_rowcol_depth_grid, 2), 1);
    loa_feat_param.GS_KGFv01_s_rowcoldepth_D_vector     = reshape(GS_KGFv01_s_rowcol_depth_D_grid, size(GS_KGFv01_s_rowcol_depth_D_grid, 1)*size(GS_KGFv01_s_rowcol_depth_D_grid, 2), 1);
    
    loa_feat_param.GS_KGFv01_N_loa_feat_cols_per_point_axis   = (N_GS_KGFv01_beta_grid * N_GS_KGFv01_k_grid * N_GS_KGFv01_s_grid);
end