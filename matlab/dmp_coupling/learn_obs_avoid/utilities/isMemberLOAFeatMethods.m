function [ is_member ] = isMemberLOAFeatMethods( eval_loa_feat_method, ...
                                                 loa_feat_methods )
    if (iscell(loa_feat_methods) == 1)
        is_member   = ismember(eval_loa_feat_method, cell2mat(loa_feat_methods));
    else
        is_member   = ismember(eval_loa_feat_method, loa_feat_methods);
    end
end
