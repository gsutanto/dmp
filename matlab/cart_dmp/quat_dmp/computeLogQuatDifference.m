function [ result ] = computeLogQuatDifference( quat_p, quat_q, varargin )
    % Author        : Giovanni Sutanto
    % Date          : December 2016
    if (nargin > 2)
        is_standardizing_quat_diff  = varargin{1};
    else
        is_standardizing_quat_diff  = 1;    % default is standardizing before applying Log Mapping
    end
    
    if (is_standardizing_quat_diff == 0)        % if NOT standardizing before applying Log Mapping
        result  = computeLogMapQuat(computeQuatProduct(...
                                            normalizeQuaternion(quat_p), ...
                                            computeQuatConjugate(quat_q)));
    else % if (is_standardizing_quat_diff ~= 0) % if standardizing before applying Log Mapping
        result  = computeLogMapQuat(standardizeNormalizeQuaternion(...
                                            computeQuatProduct(...
                                                normalizeQuaternion(quat_p), ...
                                                computeQuatConjugate(quat_q))));
    end
end
