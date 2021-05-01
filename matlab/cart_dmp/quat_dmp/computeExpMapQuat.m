function [ exp_mapped_quat ] = computeExpMapQuat( log_mapped_quat_input )
    % Author        : Giovanni Sutanto
    % Date          : December 2016
    assert(size(log_mapped_quat_input, 1) == 3, 'Each column of log_mapped_quat_input must be 3-dimensional!');

    r               = 0.5 * log_mapped_quat_input;

    len_quat        = size(r, 2);
    
    norm_r          = sqrt(sum((r .^ 2), 1));
    
    zero_norm_r_idx = find(norm_r == 0);
%     if (~isempty(zero_norm_r_idx))
%         fprintf('There will be some resulting Quaternion==[1,0,0,0]^T\n');
%     end
    non_zero_norm_r_idx = setdiff([1:len_quat], zero_norm_r_idx);
    
    % for (norm(r) == 0) case:
    exp_mapped_quat(1, zero_norm_r_idx)     = ones(1, length(zero_norm_r_idx));
    exp_mapped_quat(2:4, zero_norm_r_idx)   = zeros(3, length(zero_norm_r_idx));
    
    % for (norm(r) != 0) case:
    exp_mapped_quat(1, non_zero_norm_r_idx)     = cos(norm_r(1, non_zero_norm_r_idx));
    exp_mapped_quat(2:4, non_zero_norm_r_idx)   = repmat(exp(log(sin(norm_r(1, non_zero_norm_r_idx))) - ...
                                                             log(norm_r(1, non_zero_norm_r_idx))), 3, 1)...
                                                  .* r(:, non_zero_norm_r_idx);
    
    % there is possibility that the computed exp_mapped_quat is an
    % imaginary number, therefore we check this:
    if (max(abs(imag(exp_mapped_quat))) > 1.0e-10)
        keyboard;   % perform manual inspection if necessary ...
    end
    
    % take the real components only:
    exp_mapped_quat     = real(exp_mapped_quat);
    
    % don't forget to normalize the resulting Quaternion:
    exp_mapped_quat     = normalizeQuaternion(exp_mapped_quat);
end
