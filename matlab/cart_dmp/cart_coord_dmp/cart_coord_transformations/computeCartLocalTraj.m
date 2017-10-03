function [ varargout ] = computeCartLocalTraj( varargin )
    % Author: Giovanni Sutanto
    % Date  : August 02, 2016
    
    cart_global_traj        = varargin{1};
    if (nargin > 1)
        is_plot_trajs       = varargin{2};
    else
        is_plot_trajs       = 0;
    end
    if (nargin > 2)
        ctraj_local_coordinate_frame_selection  = varargin{3};
    else
        ctraj_local_coordinate_frame_selection  = 1;    % default is gsutanto's ctraj_local_coordinate_frame_selection
    end
    
    if (iscell(cart_global_traj) == 1)
        cart_global_xyz = cart_global_traj{1,1}.';
    else
        cart_global_xyz = cart_global_traj.';
    end
    
    [ T_local_to_global_H, T_global_to_local_H ] = computeCTrajCoordTransforms( cart_global_traj, ctraj_local_coordinate_frame_selection );
    
    [ cart_local_traj ] = convertCTrajAtOldToNewCoordSys( cart_global_traj, T_global_to_local_H );
    
    translation_vec     = T_local_to_global_H(1:3,4);
    rotation_mat        = T_local_to_global_H(1:3,1:3);
    local_x_axis        = rotation_mat(:,1);
    local_y_axis        = rotation_mat(:,2);
    local_z_axis        = rotation_mat(:,3);
    
    if (iscell(cart_local_traj) == 1)
        cart_local_xyz_H= [cart_local_traj{1,1}.'; ones(1,size(cart_local_traj{1,1},1))];
    else
        cart_local_xyz_H= [cart_local_traj.'; ones(1,size(cart_local_traj,1))];
    end
    
    if (is_plot_trajs)
        figure;
        axis            equal;
        hold            on;
            px1         = quiver3(0,0,0,1,0,0,'r');
            py1         = quiver3(0,0,0,0,1,0,'g');
            pz1         = quiver3(0,0,0,0,0,1,'b');
            ptraj1      = plot3(cart_local_xyz_H(1,:)', cart_local_xyz_H(2,:)', cart_local_xyz_H(3,:)', 'c', 'LineWidth', 3);
            legend([px1, py1, pz1, ptraj1], 'Local x-axis', 'Local y-axis', 'Local z-axis', 'Trajectory Representation in Local Coordinate System');
        hold            off;

        reproduced_cart_global_xyz_H    = T_local_to_global_H * cart_local_xyz_H;
        figure;
        axis            equal;
        hold            on;
            px2         = quiver3(translation_vec(1,1),translation_vec(2,1),translation_vec(3,1),local_x_axis(1,1),local_x_axis(2,1),local_x_axis(3,1),'r');
            py2       	= quiver3(translation_vec(1,1),translation_vec(2,1),translation_vec(3,1),local_y_axis(1,1),local_y_axis(2,1),local_y_axis(3,1),'g');
            pz2     	= quiver3(translation_vec(1,1),translation_vec(2,1),translation_vec(3,1),local_z_axis(1,1),local_z_axis(2,1),local_z_axis(3,1),'b');
            ptraj2    	= plot3(cart_global_xyz(1,:)', cart_global_xyz(2,:)', cart_global_xyz(3,:)', 'cx', reproduced_cart_global_xyz_H(1,:)', reproduced_cart_global_xyz_H(2,:)', reproduced_cart_global_xyz_H(3,:)', 'm+');
            legend([px2, py2, pz2], 'Local x-axis', 'Local y-axis', 'Local z-axis');
        hold            off;

        figure;
        axis            equal;
        hold            on;
            px1             = quiver3(0,0,0,1,0,0,'r');
            py1             = quiver3(0,0,0,0,1,0,'g');
            pz1             = quiver3(0,0,0,0,0,1,'b');
            px2             = quiver3(translation_vec(1,1),translation_vec(2,1),translation_vec(3,1),local_x_axis(1,1),local_x_axis(2,1),local_x_axis(3,1),'m-.', 'LineWidth', 2);
            py2             = quiver3(translation_vec(1,1),translation_vec(2,1),translation_vec(3,1),local_y_axis(1,1),local_y_axis(2,1),local_y_axis(3,1),'y-.', 'LineWidth', 2);
            pz2             = quiver3(translation_vec(1,1),translation_vec(2,1),translation_vec(3,1),local_z_axis(1,1),local_z_axis(2,1),local_z_axis(3,1),'c-.', 'LineWidth', 2);
            ptraj2          = plot3(cart_global_xyz(1,:)', cart_global_xyz(2,:)', cart_global_xyz(3,:)', 'k', 'LineWidth', 3);
            legend([px1, py1, pz1, px2, py2, pz2, ptraj2], 'Global x-axis', 'Global y-axis', 'Global z-axis', 'Local x-axis', 'Local y-axis', 'Local z-axis',...
                                                           'Trajectory');
            title('Global and Local Coordinate System in DMP');
        hold            off;
    end
    
    varargout(1)    = {cart_local_traj};
    varargout(2)    = {T_local_to_global_H};
    varargout(3)    = {T_global_to_local_H};
end