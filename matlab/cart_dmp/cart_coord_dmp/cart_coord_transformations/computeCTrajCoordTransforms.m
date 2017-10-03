function [ varargout ] = computeCTrajCoordTransforms( varargin )
    % Author     : Giovanni Sutanto
    % Date       : August 04, 2016
    % Description: Computes homogeneous transformation matrices 
    %              between the trajectory's local and global coordinate systems.
    
    cart_global_traj                            = varargin{1};
    if (nargin > 1)
        ctraj_local_coordinate_frame_selection  = varargin{2};
    else
        ctraj_local_coordinate_frame_selection  = 1;    % default is gsutanto's ctraj_local_coordinate_frame_selection
    end
    if (nargin > 2)
        parallel_vector_projection_thresh       = varargin{3};
    else
        parallel_vector_projection_thresh       = 0.9;
    end
    
    if (iscell(cart_global_traj) == 1)
        cart_global_xyz   	= cart_global_traj{1,1}.';
    else
        cart_global_xyz   	= cart_global_traj.';
    end
    
    start_cart_global_xyz   = cart_global_xyz(:,1);
    goal_cart_global_xyz    = cart_global_xyz(:,end);
    
    translation_vec         = start_cart_global_xyz;  % translation vector
    
    global_x_axis           = [1;0;0];
    global_y_axis           = [0;1;0];
    global_z_axis           = [0;0;1];
    
    local_x_axis            = goal_cart_global_xyz - start_cart_global_xyz;
    local_x_axis            = local_x_axis/(norm(local_x_axis));    % normalize the local x-axis
    
    if (ctraj_local_coordinate_frame_selection == 0)        % no coordinate transformation
        local_x_axis        = global_x_axis;
        local_y_axis        = global_y_axis;
        local_z_axis        = global_z_axis;
        translation_vec     = [0;0;0];
    elseif (ctraj_local_coordinate_frame_selection == 1)    % gsutanto's ctraj_local_coordinate_frame_selection
        % gsutanto's comment on this ctraj_local_coordinate_frame_selection:
        % This ctraj_local_coordinate_frame_selection is designed to tackle the problem on schaal's ctraj_local_coordinate_frame_selection,
        % which at certain cases might have a discontinuous local coordinate
        % transformation as the local x-axis changes with respect to the
        % global coordinate system, due to the use of hard-coded IF
        % statement to tackle the situation where local x-axis gets close
        % as being parallel to global z-axis. gsutanto's ctraj_local_coordinate_frame_selection tackle this
        % problem by defining 5 basis vectors as candidates of anchor local
        % z-axis at 5 different situations as defined below. To ensure
        % continuity/smoothness on transitions among these 5 cases, 
        % we introduce a (normalized) weighted linear combination of these 
        % 5 basis vectors (the weights are gaussian kernels centered at the
        % 5 different situations/cases).
        cos_theta           = global_z_axis.' * local_x_axis;
        theta               = acos(cos_theta);
        
        % compute the kernel centers and spread/standard deviation (D):
        theta_kernel_centers= [0:pi/4:pi];
        theta_kernel_Ds     = (diff(theta_kernel_centers)*0.55).^2;
        theta_kernel_Ds     = 1./[theta_kernel_Ds, theta_kernel_Ds(end)];
        
        % compute the kernel/basis vector weights:
        basis_vector_weights_unnormalized   = exp(-0.5*((theta-theta_kernel_centers).^2).*theta_kernel_Ds);
        
        % columns of basis_matrix below are the basis vectors:
        basis_matrix        = zeros(3, length(theta_kernel_centers));
        % theta == 0 (local_x_axis is perfectly aligned with global_z_axis)
        % corresponds to anchor_local_z_axis = global_y_axis:
        basis_matrix(:,1)   = global_y_axis;
        % theta == pi/4 corresponds to 
        % anchor_local_z_axis = global_z_axis X local_x_axis:
        basis_matrix(:,2)   = cross(global_z_axis, local_x_axis);
        % theta == pi/2 (local_x_axis is perpendicular from global_z_axis)
        % corresponds to anchor_local_z_axis = global_z_axis:
        basis_matrix(:,3)   = global_z_axis;
        % theta == 3*pi/4 corresponds to 
        % anchor_local_z_axis = -global_z_axis X local_x_axis:
        basis_matrix(:,4)   = -cross(global_z_axis, local_x_axis);
        % theta == pi (local_x_axis is perfectly aligned with -global_z_axis)
        % corresponds to anchor_local_z_axis = -global_y_axis:
        basis_matrix(:,5)   = -global_y_axis;
        
        % anchor_local_z_axis are the normalized weighted combination of
        % the basis vectors:
        anchor_local_z_axis = basis_matrix * basis_vector_weights_unnormalized.'/sum(basis_vector_weights_unnormalized);
        anchor_local_z_axis = anchor_local_z_axis/(norm(anchor_local_z_axis));  % normalize the anchor_local_z_axis
        
        local_y_axis        = cross(anchor_local_z_axis, local_x_axis);
        local_y_axis        = local_y_axis/(norm(local_y_axis));    % normalize local y-axis
        local_z_axis        = cross(local_x_axis, local_y_axis);
        local_z_axis        = local_z_axis/(norm(local_z_axis));    % normalize the real local z-axis
    elseif (ctraj_local_coordinate_frame_selection == 2)    % schaal's ctraj_local_coordinate_frame_selection
        if (abs(global_z_axis.' * local_x_axis) <= parallel_vector_projection_thresh)   % check if local x-axis is "too parallel" with the global z-axis; if not:
            local_z_axis    = global_z_axis;                        % dummy local z-axis
            local_y_axis    = cross(local_z_axis, local_x_axis);
            local_y_axis    = local_y_axis/(norm(local_y_axis));    % normalize local y-axis
            local_z_axis    = cross(local_x_axis, local_y_axis);
            local_z_axis    = local_z_axis/(norm(local_z_axis));    % normalize the real local z-axis
        else                    % if it is "parallel enough":
            local_y_axis    = global_y_axis;                        % dummy local y-axis
            local_z_axis    = cross(local_x_axis, local_y_axis);
            local_z_axis    = local_z_axis/(norm(local_z_axis));    % normalize local z-axis
            local_y_axis    = cross(local_z_axis, local_x_axis);
            local_y_axis    = local_y_axis/(norm(local_y_axis));    % normalize the real local y-axis
        end
    end
    
    rotation_mat        = [local_x_axis, local_y_axis, local_z_axis];
    
    % relative homogeneous transformation matrix from local to global 
    % coordinate system:
    T_local_to_global_H = [[rotation_mat,translation_vec]; [0,0,0,1]];
    T_global_to_local_H = [[rotation_mat.',-(rotation_mat.')*translation_vec]; [0,0,0,1]];
    
    varargout(1)        = {T_local_to_global_H};
    varargout(2)        = {T_global_to_local_H};
end

