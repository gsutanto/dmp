function [ quat_output ] = standardizeNormalizeQuaternion( quat_input )
    % Author        : Giovanni Sutanto
    % Date          : December 2016
    
    quat_output         = quat_input;
    
    % Standardize (make sure that unique Quaternion represents 
    % unique orientation)
    quat_idx_tobe_std   = find(quat_output(1,:) < 0);
    if (~isempty(quat_idx_tobe_std))
%         fprintf('Standardizing some Quaternions for uniqueness ...\n');
        quat_output(:,quat_idx_tobe_std)= -quat_output(:,quat_idx_tobe_std);
    end
    
    quat_output         = normalizeQuaternion(quat_output);
end

