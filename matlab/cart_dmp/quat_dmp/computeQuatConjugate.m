function [ quat_output_traj ] = computeQuatConjugate( quat_input_traj )
    quat_output_traj(1,:)   = quat_input_traj(1,:);
    quat_output_traj(2:4,:) = -quat_input_traj(2:4,:);
    
    quat_output_traj    = normalizeQuaternion(quat_output_traj);
end