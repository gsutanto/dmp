function [ varargout ] = computeObsAvoidCtFeatPerPoint( varargin )
    loa_feat_param  = varargin{1};
    endeff_state    = varargin{2};
    obs_state       = varargin{3};
    tau             = varargin{4}/0.5; % tau is relative to 0.5 seconds (similar to tau computed in dcp_franzi.m)
    loa_feat_methods= varargin{5};
    if (nargin > 5)
        px          = varargin{6};  % phase variable's x
        pv          = varargin{7};  % phase variable's v
    else
        px          = 0;
        pv          = 0;
    end
    
    if (iscell(loa_feat_methods) == 1)
        N_loa_feat_methods      = length(loa_feat_methods);
        
        loa_feat_matrix_per_point_cell  = cell(1,N_loa_feat_methods);
        
        for i=1:N_loa_feat_methods
            
            loa_feat_matrix_per_point_cell{1,i} = computeObsAvoidCtFeatPerPoint( loa_feat_param, ...
                                                                                 endeff_state, ...
                                                                                 obs_state, ...
                                                                                 (2.0 * tau), ...
                                                                                 loa_feat_methods{1,i}, ...
                                                                                 px, ...
                                                                                 pv );
            
        end
        
        loa_feat_matrix_per_point   = cell2mat(loa_feat_matrix_per_point_cell);
        
        varargout(1) 	= {loa_feat_matrix_per_point};
    else
        
        loa_feat_method = loa_feat_methods;
        
        if ((loa_feat_method == 0) || (loa_feat_method == 5)) % Akshara's Humanoids'14 features
            varargout   = {computeAksharaHumanoids2014ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau )};
        elseif (loa_feat_method == 1) % Potential Field 2nd Dynamic Obst Avoid features
            varargout   = {computePF_DYN2ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau )};
        elseif (loa_feat_method == 2) % gsutanto's Kernelized General Features (KGF) version 01
            varargout   = {computeGS_KGFv01ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )};
        elseif (loa_feat_method == 3) % Franzi's Kernelized General Features v01
            varargout   = {computeFM_KGFv01ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )};
        elseif (loa_feat_method == 4) % Potential Field 3rd Dynamic Obst Avoid features (have some sense of KGF too...)
            varargout   = {computePF_DYN3ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )};
        elseif (loa_feat_method == 6) % Potential Field 4th Dynamic Obst Avoid features (have some sense of KGF too...)
            varargout   = {computePF_DYN4ObstAvoidCtFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau, px, pv )};
        end
        
    end
end
