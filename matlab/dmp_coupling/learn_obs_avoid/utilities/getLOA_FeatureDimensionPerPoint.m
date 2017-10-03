function [ varargout ] = getLOA_FeatureDimensionPerPoint( varargin )
    if (iscell(varargin{1}) == 1)
        D                   = varargin{2};
        loa_feat_param      = varargin{3};
        
        loa_feat_param.N_loa_feat_cols_per_point_axis   = 0;
        N_loa_feat_cols_per_point_axis_checksum         = 0;
        
        for i=1:length(varargin{1})
            loa_feat_method = varargin{1}{1,i};
            [N_loa_feat_rows_per_point_axis, N_loa_feat_subcols_per_point_axis, loa_feat_param] = getLOA_FeatureDimensionPerPoint( loa_feat_method, D, loa_feat_param );
            N_loa_feat_cols_per_point_axis_checksum     = N_loa_feat_cols_per_point_axis_checksum + N_loa_feat_subcols_per_point_axis;
        end
        
        if (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
            if (N_loa_feat_cols_per_point_axis_checksum ~= loa_feat_param.N_loa_feat_cols_per_point_axis)
                disp('ERROR: N_loa_feat_cols_per_point_axis_checksum mis-match!!!');
            end
        elseif (strcmp(loa_feat_param.feat_constraint_mode, '_UNCONSTRAINED_') == 1)
            if (N_loa_feat_cols_per_point_axis_checksum ~= D * loa_feat_param.N_loa_feat_cols_per_point_axis)
                disp('ERROR: N_loa_feat_cols_per_point_axis_checksum mis-match!!!');
            end
        end
        
        varargout(1)	= {N_loa_feat_rows_per_point_axis};
        varargout(2)    = {N_loa_feat_cols_per_point_axis_checksum};
        varargout(3)    = {loa_feat_param};
    else
        loa_feat_method = varargin{1};
        
        D             	= varargin{2};
        loa_feat_param  = varargin{3};

        if (nargout == 1) % deprecated option; in the future we will use the 3 output arguments form

            if (loa_feat_method == 0) % Akshara's Humanoids'14 features
                N_loa_feat_vect_per_point_axis  = D*loa_feat_param.AF_H14_N_loa_feat_cols_per_point_axis;
            elseif (loa_feat_method == 1) % Potential Field 2nd Dynamic Obst Avoid features
                N_loa_feat_vect_per_point_axis  = D*loa_feat_param.PF_DYN2_N_loa_feat_cols_per_point_axis;
            end

            varargout(1)    = {N_loa_feat_vect_per_point_axis};

        elseif (nargout == 3)
            
            if ((loa_feat_method == 0) || (loa_feat_method == 5)) % Akshara's Humanoids'14 features
                N_loa_feat_cols_per_point_axis  = loa_feat_param.AF_H14_N_loa_feat_cols_per_point_axis;
                
                loa_feat_param.AF_H14_feat_start_col_idx    = loa_feat_param.N_loa_feat_cols_per_point_axis + 1;
                loa_feat_param.AF_H14_feat_end_col_idx      = loa_feat_param.AF_H14_feat_start_col_idx + ...
                                                              loa_feat_param.AF_H14_N_loa_feat_cols_per_point_axis - 1;
                loa_feat_param.N_loa_feat_cols_per_point_axis   = loa_feat_param.AF_H14_feat_end_col_idx;
            elseif (loa_feat_method == 1) % Potential Field 2nd Dynamic Obst Avoid features
                N_loa_feat_cols_per_point_axis  = loa_feat_param.PF_DYN2_N_loa_feat_cols_per_point_axis;
    
                loa_feat_param.PF_DYN2_feat_start_col_idx   = loa_feat_param.N_loa_feat_cols_per_point_axis + 1;
                loa_feat_param.PF_DYN2_feat_end_col_idx     = loa_feat_param.PF_DYN2_feat_start_col_idx + ...
                                                              loa_feat_param.PF_DYN2_N_loa_feat_cols_per_point_axis - 1;
                loa_feat_param.N_loa_feat_cols_per_point_axis   = loa_feat_param.PF_DYN2_feat_end_col_idx;
            elseif (loa_feat_method == 2) % gsutanto's Kernelized General Features (KGF) version 01 
                N_loa_feat_cols_per_point_axis  = loa_feat_param.GS_KGFv01_N_loa_feat_cols_per_point_axis;
                
                loa_feat_param.GS_KGFv01_feat_start_col_idx = loa_feat_param.N_loa_feat_cols_per_point_axis + 1;
                loa_feat_param.GS_KGFv01_feat_end_col_idx   = loa_feat_param.GS_KGFv01_feat_start_col_idx + ...
                                                              loa_feat_param.GS_KGFv01_N_loa_feat_cols_per_point_axis - 1;
                loa_feat_param.N_loa_feat_cols_per_point_axis   = loa_feat_param.GS_KGFv01_feat_end_col_idx;
            elseif (loa_feat_method == 3) % Franzi's Kernelized General Features v01
                N_loa_feat_cols_per_point_axis  = loa_feat_param.FM_KGFv01_N_loa_feat_cols_per_point_axis;
                
                loa_feat_param.FM_KGFv01_feat_start_col_idx = loa_feat_param.N_loa_feat_cols_per_point_axis + 1;
                loa_feat_param.FM_KGFv01_feat_end_col_idx   = loa_feat_param.FM_KGFv01_feat_start_col_idx + ...
                                                              loa_feat_param.FM_KGFv01_N_loa_feat_cols_per_point_axis - 1;
                loa_feat_param.N_loa_feat_cols_per_point_axis   = loa_feat_param.FM_KGFv01_feat_end_col_idx;
            elseif (loa_feat_method == 4) % Potential Field 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
                N_loa_feat_cols_per_point_axis  = loa_feat_param.PF_DYN3_N_loa_feat_cols_per_point_axis;
                
                loa_feat_param.PF_DYN3_feat_start_col_idx   = loa_feat_param.N_loa_feat_cols_per_point_axis + 1;
                loa_feat_param.PF_DYN3_feat_end_col_idx     = loa_feat_param.PF_DYN3_feat_start_col_idx + ...
                                                              loa_feat_param.PF_DYN3_N_loa_feat_cols_per_point_axis - 1;
                loa_feat_param.N_loa_feat_cols_per_point_axis   = loa_feat_param.PF_DYN3_feat_end_col_idx;
            elseif (loa_feat_method == 6) % Potential Field 4th Dynamic Obst Avoid features (have some sense of KGF too...)
                N_loa_feat_cols_per_point_axis  = loa_feat_param.PF_DYN4_N_loa_feat_cols_per_point_axis;
                
                loa_feat_param.PF_DYN4_feat_start_col_idx   = loa_feat_param.N_loa_feat_cols_per_point_axis + 1;
                loa_feat_param.PF_DYN4_feat_end_col_idx     = loa_feat_param.PF_DYN4_feat_start_col_idx + ...
                                                              loa_feat_param.PF_DYN4_N_loa_feat_cols_per_point_axis - 1;
                loa_feat_param.N_loa_feat_cols_per_point_axis   = loa_feat_param.PF_DYN4_feat_end_col_idx;
            elseif (loa_feat_method == 7) % Neural Network
                N_loa_feat_cols_per_point_axis  = loa_feat_param.NN_N_loa_feat_cols_per_point_axis;
                
%                 loa_feat_param.NN_feat_start_col_idx        = loa_feat_param.N_loa_feat_cols_per_point_axis + 1;
%                 loa_feat_param.NN_feat_end_col_idx          = loa_feat_param.NN_feat_start_col_idx + ...
%                                                               loa_feat_param.NN_N_loa_feat_cols_per_point_axis - 1;
                loa_feat_param.N_loa_feat_cols_per_point_axis   = loa_feat_param.NN_N_loa_feat_cols_per_point_axis;
            end

            if (strcmp(loa_feat_param.feat_constraint_mode, '_UNCONSTRAINED_') == 1)
                N_loa_feat_rows_per_point_axis      = 1;
                
                if ((loa_feat_method ~= 3) && (loa_feat_method ~= 7)) % except Franzi's KGF and NN
                    N_loa_feat_cols_per_point_axis  = D*N_loa_feat_cols_per_point_axis;
                end
            elseif (strcmp(loa_feat_param.feat_constraint_mode, '_CONSTRAINED_') == 1)
                N_loa_feat_rows_per_point_axis      = D;
            end

            varargout(1)    = {N_loa_feat_rows_per_point_axis};
            varargout(2)    = {N_loa_feat_cols_per_point_axis};
            varargout(3)    = {loa_feat_param};

        end
    end
end