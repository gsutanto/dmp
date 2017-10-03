function [ quat_output ] = normalizeQuaternion( quat_input )
    % Author        : Giovanni Sutanto
    % Date          : February 2017
    
    quat_output         = quat_input;
    
    % Normalize (make sure that norm(Quaternion) == 1)
    quat_norm   = sqrt(sum((quat_output .^ 2), 1));
    quat_output = quat_output ./ repmat(quat_norm, 4 , 1);
end

