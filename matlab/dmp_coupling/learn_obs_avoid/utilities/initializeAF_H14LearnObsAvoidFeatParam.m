function [ loa_feat_param ] = initializeAF_H14LearnObsAvoidFeatParam( varargin )
    loa_feat_param                      = varargin{1};
    N_AF_H14_beta_phi1_phi2_grid        = varargin{2};
    AF_H14_beta_phi1_phi2_low           = varargin{3};
    AF_H14_beta_phi1_phi2_high          = varargin{4};
    N_AF_H14_k_phi1_phi2_grid           = varargin{5};
    AF_H14_k_phi1_phi2_low              = varargin{6};
    AF_H14_k_phi1_phi2_high             = varargin{7};
    if (nargin > 7)
        N_AF_H14_k_phi3_grid            = varargin{8};
        AF_H14_k_phi3_low               = varargin{9};
        AF_H14_k_phi3_high              = varargin{10};
    else
        N_AF_H14_k_phi3_grid            = 0;
        AF_H14_k_phi3_low               = 0;
        AF_H14_k_phi3_high              = 0;
    end
    if (nargin > 10)
        k_grid_mode                     = varargin{11};
    else
        k_grid_mode                     = '_LINEAR_';
    end
    if (nargin > 11)
        loa_feat_param.point_feat_mode  = varargin{12};
    else
        loa_feat_param.point_feat_mode 	= '_SUM_OBS_POINTS_FEATURE_CONTRIBUTION_';
    end
    if (nargin > 12)
        loa_feat_param.feat_constraint_mode = varargin{13};
    else
        loa_feat_param.feat_constraint_mode = '_UNCONSTRAINED_';
    end
    
    loa_feat_param.AF_H14_N_beta_phi1_phi2_grid = N_AF_H14_beta_phi1_phi2_grid;
    loa_feat_param.AF_H14_N_k_phi1_phi2_grid    = N_AF_H14_k_phi1_phi2_grid;
    loa_feat_param.AF_H14_N_k_phi3_grid         = N_AF_H14_k_phi3_grid;
    
    [AF_H14_beta_phi1_phi2_rect_grid, AF_H14_k_phi1_phi2_rect_grid] = meshgrid(linspace(AF_H14_beta_phi1_phi2_low, AF_H14_beta_phi1_phi2_high, N_AF_H14_beta_phi1_phi2_grid),...
                                                                               linspace(AF_H14_k_phi1_phi2_low, AF_H14_k_phi1_phi2_high, N_AF_H14_k_phi1_phi2_grid));
    loa_feat_param.AF_H14_beta_phi1_phi2_vector = reshape(AF_H14_beta_phi1_phi2_rect_grid, N_AF_H14_beta_phi1_phi2_grid*N_AF_H14_k_phi1_phi2_grid, 1);
    if (strcmp(k_grid_mode, '_LINEAR_') == 1)
        loa_feat_param.AF_H14_k_phi1_phi2_vector= reshape(AF_H14_k_phi1_phi2_rect_grid, N_AF_H14_beta_phi1_phi2_grid*N_AF_H14_k_phi1_phi2_grid, 1);
    elseif (strcmp(k_grid_mode, '_QUADRATIC_') == 1)
        loa_feat_param.AF_H14_k_phi1_phi2_vector= reshape((AF_H14_k_phi1_phi2_rect_grid.^2), N_AF_H14_beta_phi1_phi2_grid*N_AF_H14_k_phi1_phi2_grid, 1);
    elseif (strcmp(k_grid_mode, '_INVERSE_QUADRATIC_') == 1)
        loa_feat_param.AF_H14_k_phi1_phi2_vector= reshape((1.0./(AF_H14_k_phi1_phi2_rect_grid.^2)), N_AF_H14_beta_phi1_phi2_grid*N_AF_H14_k_phi1_phi2_grid, 1);
    end
    
    if (loa_feat_param.AF_H14_N_k_phi3_grid > 0)
        if (strcmp(k_grid_mode, '_LINEAR_') == 1)
            loa_feat_param.AF_H14_k_phi3_vector = (linspace(AF_H14_k_phi3_low, AF_H14_k_phi3_high, N_AF_H14_k_phi3_grid)).';
        elseif (strcmp(k_grid_mode, '_QUADRATIC_') == 1)
            loa_feat_param.AF_H14_k_phi3_vector = ((linspace(AF_H14_k_phi3_low, AF_H14_k_phi3_high, N_AF_H14_k_phi3_grid)).^2).';
        elseif (strcmp(k_grid_mode, '_INVERSE_QUADRATIC_') == 1)
            loa_feat_param.AF_H14_k_phi3_vector = (1.0./((linspace(AF_H14_k_phi3_low, AF_H14_k_phi3_high, N_AF_H14_k_phi3_grid)).^2)).';
        end
    end
    
    loa_feat_param.AF_H14_N_loa_feat_cols_per_point_axis    = ((loa_feat_param.AF_H14_N_beta_phi1_phi2_grid * loa_feat_param.AF_H14_N_k_phi1_phi2_grid)+loa_feat_param.AF_H14_N_k_phi3_grid);
end