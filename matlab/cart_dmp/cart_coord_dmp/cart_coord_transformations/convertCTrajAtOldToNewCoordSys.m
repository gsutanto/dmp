function [ cart_traj_new_coord_sys ] = convertCTrajAtOldToNewCoordSys( cart_traj_old_coord_sys, relative_homogeneous_transformation_matrix_old_to_new )
    % Author     : Giovanni Sutanto
    % Date       : August 04, 2016
    % Description: Converts Cartesian trajectory
    %              from its representation in the old coordinate system 
    %              to its representation in the new coordinate system
    
    if (iscell(cart_traj_old_coord_sys) == 1)
        is_processing_states    = 1;
    else
        is_processing_states    = 0;    % processing coordinates only
    end
    
    if (is_processing_states)
        cart_old_xyz        = cart_traj_old_coord_sys{1,1}.';
        cart_old_xdydzd     = cart_traj_old_coord_sys{2,1}.';
        cart_old_xddyddzdd  = cart_traj_old_coord_sys{3,1}.';
    else
        cart_old_xyz        = cart_traj_old_coord_sys.';
    end
    
    T_old_to_new_H      = relative_homogeneous_transformation_matrix_old_to_new;
    R_old_to_new        = T_old_to_new_H(1:3,1:3);
    
    cart_old_xyz_H      = [cart_old_xyz; ones(1, size(cart_old_xyz,2))];
    cart_new_xyz_H      = T_old_to_new_H * cart_old_xyz_H;
    
    if (is_processing_states)
        cart_new_xdydzd     = R_old_to_new * cart_old_xdydzd;

        cart_new_xddyddzdd  = R_old_to_new * cart_old_xddyddzdd;

        cart_traj_new_coord_sys{1,1}    = cart_new_xyz_H(1:3,:).';
        cart_traj_new_coord_sys{2,1}    = cart_new_xdydzd.';
        cart_traj_new_coord_sys{3,1}    = cart_new_xddyddzdd.';
    else
        cart_traj_new_coord_sys         = cart_new_xyz_H(1:3,:).';
    end
end