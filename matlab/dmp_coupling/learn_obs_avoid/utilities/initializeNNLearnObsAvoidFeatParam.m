function [ loa_feat_param ] = initializeNNLearnObsAvoidFeatParam( varargin )
    loa_feat_param  = varargin{1};
    NN_net_struct   = varargin{2};
    NN_N_feats     	= varargin{3};  % number of features for Neural Network

    loa_feat_param.feat_constraint_mode                 = '_UNCONSTRAINED_';
    loa_feat_param.NN_net_struct                        = NN_net_struct;
    loa_feat_param.NN_N_loa_feat_cols_per_point_axis    = NN_N_feats;
end