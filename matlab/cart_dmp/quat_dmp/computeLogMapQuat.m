function [ log_mapped_quat ] = computeLogMapQuat( quat_input )
    % Author        : Giovanni Sutanto
    % Date          : December 2016
    assert(size(quat_input, 1) == 4, 'Each column of quat_input must be a Quaternion (4-dimensional)!');
    
    len_quat        = size(quat_input, 2);
    
    % normalize the input Quaternion first:
    quat_prep       = normalizeQuaternion(quat_input);
    
    u               = quat_prep(1,:);
    q               = quat_prep(2:4,:);
    
    arccos_u        = acos(u);
    sin_arccos_u    = sin(arccos_u);
    origin_quat_idx = find(sin_arccos_u == 0);
%     if (~isempty(origin_quat_idx))
%         fprintf('There are some Quaternion==[1,0,0,0]^T\n');
%     end
    non_origin_quat_idx = setdiff([1:len_quat], origin_quat_idx);
    log_non_origin_quat_multiplier  = log(arccos_u(1,non_origin_quat_idx)) - ...
                                      log(sin_arccos_u(1,non_origin_quat_idx));
    
  	log_mapped_quat = q;
    log_mapped_quat(:,non_origin_quat_idx) = exp(log(q(:,non_origin_quat_idx)) + ...
                                                 repmat(log_non_origin_quat_multiplier, 3, 1));
    
    % there is possibility that the computed log_mapped_quat is a
    % complex number, therefore we check this:
    if (max(max(abs(imag(log_mapped_quat)))) > 1.0e-10)
        keyboard;   % perform manual inspection if necessary ...
    end
    
    % take the real components only:
    log_mapped_quat     = 2.0 * real(log_mapped_quat);
end
