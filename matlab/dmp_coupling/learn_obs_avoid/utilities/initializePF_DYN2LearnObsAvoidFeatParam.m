function [ loa_feat_param ] = initializePF_DYN2LearnObsAvoidFeatParam( varargin )
    loa_feat_param                      = varargin{1};
    N_PF_DYN2_beta_grid                 = varargin{2};
    PF_DYN2_beta_low                    = varargin{3};
    PF_DYN2_beta_high                   = varargin{4};
    N_PF_DYN2_k_grid                    = varargin{5};
    PF_DYN2_k_low                       = varargin{6};
    PF_DYN2_k_high                      = varargin{7};
    if (nargin > 7)
        k_grid_mode                     = varargin{8};
    else
        k_grid_mode                     = '_LINEAR_';
    end
    if (nargin > 8)
        loa_feat_param.point_feat_mode  = varargin{9};
    else
        loa_feat_param.point_feat_mode 	= '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_';
    end
    if (nargin > 9)
        loa_feat_param.feat_constraint_mode = varargin{10};
    else
        loa_feat_param.feat_constraint_mode = '_UNCONSTRAINED_';
    end
    
    loa_feat_param.PF_DYN2_N_beta_grid  = N_PF_DYN2_beta_grid;
    loa_feat_param.PF_DYN2_N_k_grid     = N_PF_DYN2_k_grid;
    
    [PF_DYN2_beta_rect_grid, PF_DYN2_k_rect_grid] = meshgrid(linspace(PF_DYN2_beta_low, PF_DYN2_beta_high, N_PF_DYN2_beta_grid),...
                                                             linspace(PF_DYN2_k_low, PF_DYN2_k_high, N_PF_DYN2_k_grid));
    loa_feat_param.PF_DYN2_beta_vector  = reshape(PF_DYN2_beta_rect_grid, N_PF_DYN2_beta_grid*N_PF_DYN2_k_grid, 1);
    if (strcmp(k_grid_mode, '_LINEAR_') == 1)
        loa_feat_param.PF_DYN2_k_vector = reshape(PF_DYN2_k_rect_grid, N_PF_DYN2_beta_grid*N_PF_DYN2_k_grid, 1);
    elseif (strcmp(k_grid_mode, '_QUADRATIC_') == 1)
        loa_feat_param.PF_DYN2_k_vector = reshape((PF_DYN2_k_rect_grid.^2), N_PF_DYN2_beta_grid*N_PF_DYN2_k_grid, 1);
    elseif (strcmp(k_grid_mode, '_INVERSE_QUADRATIC_') == 1)
        loa_feat_param.PF_DYN2_k_vector = reshape((1.0./(PF_DYN2_k_rect_grid.^2)), N_PF_DYN2_beta_grid*N_PF_DYN2_k_grid, 1);
    end
    
    loa_feat_param.PF_DYN2_N_loa_feat_cols_per_point_axis   = (loa_feat_param.PF_DYN2_N_beta_grid * loa_feat_param.PF_DYN2_N_k_grid);
end