function [ loa_feat_param ] = initializePF_DYN3LearnObsAvoidFeatParam( varargin )
    loa_feat_param                          = varargin{1};
    N_PF_DYN3_beta_grid                     = varargin{2};
    PF_DYN3_beta_low                        = varargin{3};
    PF_DYN3_beta_high                       = varargin{4};
    N_PF_DYN3_k_grid                        = varargin{5};
    PF_DYN3_k_low                           = varargin{6};
    PF_DYN3_k_high                          = varargin{7};
    N_PF_DYN3_s_grid                        = varargin{8};
    loa_feat_param.c_order                  = varargin{9};
    if (nargin > 9)
        loa_feat_param.feat_constraint_mode = varargin{10};
    else
        loa_feat_param.feat_constraint_mode = '_UNCONSTRAINED_';
    end
    if (nargin > 10)
        PF_DYN3_s_low                     	= varargin{11};
        PF_DYN3_s_high                     	= varargin{12};
    else
        PF_DYN3_s_low                     	= 0.0;
        PF_DYN3_s_high                     	= 1.0;
    end
    if (nargin > 12)
        PF_DYN3_beta_default_D              = varargin{13};
    else
        PF_DYN3_beta_default_D              = 1.0;
    end
    if (nargin > 13)
        PF_DYN3_s_default_D                 = varargin{14};
    else
        PF_DYN3_s_default_D                 = 1.0;
    end
    
    loa_feat_param.PF_DYN3_N_beta_grid    = N_PF_DYN3_beta_grid;
    loa_feat_param.PF_DYN3_N_k_grid       = N_PF_DYN3_k_grid;
    loa_feat_param.PF_DYN3_N_s_grid       = N_PF_DYN3_s_grid;
    
    loa_feat_param.PF_DYN3_beta_col_grid  = linspace(PF_DYN3_beta_low, PF_DYN3_beta_high, N_PF_DYN3_beta_grid);
    PF_DYN3_k_row_grid                    = (1.0./((linspace(PF_DYN3_k_low, PF_DYN3_k_high, N_PF_DYN3_k_grid)).^2));
    [PF_DYN3_beta_row_col_grid, PF_DYN3_k_row_col_grid] = meshgrid(loa_feat_param.PF_DYN3_beta_col_grid,...
                                                                       PF_DYN3_k_row_grid);
    if (N_PF_DYN3_beta_grid > 1)
        loa_feat_param.PF_DYN3_beta_col_D_grid  = (diff(loa_feat_param.PF_DYN3_beta_col_grid)*0.55).^2;
        loa_feat_param.PF_DYN3_beta_col_D_grid  = 1./[loa_feat_param.PF_DYN3_beta_col_D_grid, loa_feat_param.PF_DYN3_beta_col_D_grid(end)];
    else
        loa_feat_param.PF_DYN3_beta_col_D_grid  = PF_DYN3_beta_default_D;
    end
    PF_DYN3_beta_row_col_D_grid 	= repmat(loa_feat_param.PF_DYN3_beta_col_D_grid, N_PF_DYN3_k_grid, 1);
    
    PF_DYN3_beta_rowcol_vector      = reshape(PF_DYN3_beta_row_col_grid, N_PF_DYN3_beta_grid*N_PF_DYN3_k_grid, 1);
    PF_DYN3_beta_rowcol_D_vector 	= reshape(PF_DYN3_beta_row_col_D_grid, N_PF_DYN3_beta_grid*N_PF_DYN3_k_grid, 1);
    PF_DYN3_k_rowcol_vector     	= reshape(PF_DYN3_k_row_col_grid, N_PF_DYN3_beta_grid*N_PF_DYN3_k_grid, 1);

    if (N_PF_DYN3_s_grid > 1)
        t   = (PF_DYN3_s_low:(PF_DYN3_s_high-PF_DYN3_s_low)/(N_PF_DYN3_s_grid-1):PF_DYN3_s_high)*0.5;
    else
        t   = PF_DYN3_s_high*0.5;
    end
    
    if (loa_feat_param.c_order == 1)
        loa_feat_param.alpha_v 	= 25.0;
        loa_feat_param.beta_v 	= loa_feat_param.alpha_v/4.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = 1-(1+alpha/2*t)*exp(-alpha/2*t)
        loa_feat_param.PF_DYN3_s_depth_grid   = (1+((loa_feat_param.alpha_v/2)*t)).*exp(-(loa_feat_param.alpha_v/2)*t);
    else
        loa_feat_param.alpha_x  = 25.0/3.0;
        % the local models, spaced on a grid in time by applying the
        % anaytical solutions x(t) = exp(-alpha*t)
        loa_feat_param.PF_DYN3_s_depth_grid   = exp(-loa_feat_param.alpha_x*t);
    end
    if (N_PF_DYN3_s_grid > 1)
        loa_feat_param.PF_DYN3_s_depth_D_grid = (diff(loa_feat_param.PF_DYN3_s_depth_grid)*0.55).^2;
        loa_feat_param.PF_DYN3_s_depth_D_grid = 1./[loa_feat_param.PF_DYN3_s_depth_D_grid, loa_feat_param.PF_DYN3_s_depth_D_grid(end)];
    else
        loa_feat_param.PF_DYN3_s_depth_D_grid = PF_DYN3_s_default_D;
    end
    
    [PF_DYN3_s_rowcol_depth_grid, PF_DYN3_beta_rowcol_depth_grid] = meshgrid(loa_feat_param.PF_DYN3_s_depth_grid, PF_DYN3_beta_rowcol_vector.');
    PF_DYN3_k_rowcol_depth_grid       = repmat(PF_DYN3_k_rowcol_vector, 1, N_PF_DYN3_s_grid);
    PF_DYN3_beta_rowcol_depth_D_grid  = repmat(PF_DYN3_beta_rowcol_D_vector, 1, N_PF_DYN3_s_grid);
    PF_DYN3_s_rowcol_depth_D_grid     = repmat(loa_feat_param.PF_DYN3_s_depth_D_grid, size(PF_DYN3_beta_rowcol_vector, 1), 1);
    
    loa_feat_param.PF_DYN3_beta_rowcoldepth_vector    = reshape(PF_DYN3_beta_rowcol_depth_grid, size(PF_DYN3_beta_rowcol_depth_grid, 1)*size(PF_DYN3_beta_rowcol_depth_grid, 2), 1);
    loa_feat_param.PF_DYN3_beta_rowcoldepth_D_vector  = reshape(PF_DYN3_beta_rowcol_depth_D_grid, size(PF_DYN3_beta_rowcol_depth_D_grid, 1)*size(PF_DYN3_beta_rowcol_depth_D_grid, 2), 1);
    loa_feat_param.PF_DYN3_k_rowcoldepth_vector       = reshape(PF_DYN3_k_rowcol_depth_grid, size(PF_DYN3_k_rowcol_depth_grid, 1)*size(PF_DYN3_k_rowcol_depth_grid, 2), 1);
    loa_feat_param.PF_DYN3_s_rowcoldepth_vector       = reshape(PF_DYN3_s_rowcol_depth_grid, size(PF_DYN3_s_rowcol_depth_grid, 1)*size(PF_DYN3_s_rowcol_depth_grid, 2), 1);
    loa_feat_param.PF_DYN3_s_rowcoldepth_D_vector     = reshape(PF_DYN3_s_rowcol_depth_D_grid, size(PF_DYN3_s_rowcol_depth_D_grid, 1)*size(PF_DYN3_s_rowcol_depth_D_grid, 2), 1);
    
    loa_feat_param.PF_DYN3_N_loa_feat_cols_per_point_axis   = (N_PF_DYN3_beta_grid * N_PF_DYN3_k_grid * N_PF_DYN3_s_grid);
end