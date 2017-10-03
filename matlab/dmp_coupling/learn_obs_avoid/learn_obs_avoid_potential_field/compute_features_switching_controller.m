function [phi] = compute_features_switching_controller(obstavoid_feat_matrix, baseline_forcing_term, Y, Yd, k, obs3, object)

% Computing features for a ellipsoid - centre is in obs3 - average from
% Vicon markers
% Nearest points is the point on the surface, along the line joining the
% center and the end-effector

the=0;
the_o=0;

r_obs_cyl = 0.03;
r_obs_sph = 0.05;
r_obs_ell = [0.09 0.045 0.045];

%radius of the ellipsoid - pre measured

r=[0.09 0.045 0.045];

alpha   = [];

% Y contains the end-effector locations
% Yd is the velocity
%obs3 is the x-y-z location of the obstacle
for j=1:size(Y,1)
    y = Y(j,:);
    yd = Yd(j,:);
    
    if(strcmp(object, 'cyl'))
        % For each location of the end-effector, there can be four extreme
        % points on the circumference of the cylinder
        % calculate distance to all four, and then check for the minimum
        % shouldn't need to do this if I knew the closest point

        d(1)=norm([r_obs_cyl-obs3(1),0-obs3(2),y(3)]-y);
        d(2)=norm([0-obs3(1),r_obs_cyl-obs3(2),y(3)]-y);
        d(3)=norm([-r_obs_cyl-obs3(1),0-obs3(2),y(3)]-y);
        d(4)=norm([0-obs3(1),-r_obs_cyl-obs3(2),y(3)]-y);

        [d,i]=min(d);

        %depending on which of the points was closer, calculating the
        %coordinates of this point in op

        op=[r_obs_cyl*cos((pi/2)*(i-1))-obs3(1), r_obs_cyl*sin((i-1)*pi/2)-obs3(2), y(3)];

    elseif (strcmp(object, 'sph'))
        % line joining the point p and the center is the same slope. Using this
        % to calculate the polar angles of the nearrest point p.

        the_p=atan2((-y(2)+obs3(2)),(-y(1)+obs3(1)));
        phi_p=acos((y(3)-obs3(3))/(norm(y-obs3)));

        %calculating the 3D location
        op=obs3-[(r_obs_sph*cos(the_p)*sin(phi_p)) (r_obs_sph*sin(the_p)*sin(phi_p)) (-r_obs_sph*cos(phi_p))];
    
    elseif (strcmp(object, 'ell'))

        % line joining the point p and the center is the same slope. Using this
        % to calculate the polar angles of the nearrest point p.

        the_p=atan2((-y(2)+obs3(2)),(-y(1)+obs3(1)));
        phi_p=acos((y(3)-obs3(3))/(norm(y-obs3)));

        % distancce of p is calculated by exploting the fact it is on a
        % ellipsoid
        r_p=r_obs_ell(1)*r_obs_ell(2)*r_obs_ell(3)/sqrt((r_obs_ell(2)*r_obs_ell(3)*cos(the_p)*sin(phi_p))^2+(r_obs_ell(1)*r_obs_ell(3)*sin(the_p)*sin(phi_p))^2+(r_obs_ell(1)*r_obs_ell(2)*cos(phi_p))^2);
        %op=obs-(obs-y)*r_p/norm(obs-y)
        % Now we can find the global coordinates of the point P
        op=obs3-[(r_p*cos(the_p)*sin(phi_p)) (r_p*sin(the_p)*sin(phi_p)) (-r_p*cos(phi_p))];
    end
  

    % distance between point P and end-effector
%     op_r=norm(op-y);
        
    % Now calculating the features - which is the same across objects
    
    %checking that you are not at the obstacle center, and the dot products
    %are valid (so that cosine inverse doesn't return imaginary values)
    
    pc  = obs3-y;   % vector from end-effector point to the obstacle center
    pp  = op-y;     % vector from end-effector point to a closest point on the obstacle surface
    
    d_squared_vec   = [[pc*pc.'], [pp*pp.']];
    alpha_vec       = exp(-k.*k*min(d_squared_vec));
    
    alpha           = [alpha; alpha_vec];
end

    alpha_feat          = [];
    alpha_forcing_term  = [];
    for i=1:size(k,2)
        alpha_feat_temp     = repmat(alpha(:,i),[1,size(obstavoid_feat_matrix,2),size(obstavoid_feat_matrix,3)]);
        alpha_feat          = [alpha_feat, alpha_feat_temp];
        
        alpha_fterm_temp    = -[repmat(alpha(:,i),[1,1,size(baseline_forcing_term,2)])];
        alpha_forcing_term  = [alpha_forcing_term, alpha_fterm_temp];
    end
    rep_obstavoid_feat_matrix   = repmat(obstavoid_feat_matrix,[1,size(k,2)]);
    rep_baseline_forcing_term   = repmat(reshape(baseline_forcing_term,size(baseline_forcing_term,1),1,size(baseline_forcing_term,2)),[1,size(k,2)]);
    
    phi                 = [alpha_feat.*rep_obstavoid_feat_matrix, alpha_forcing_term.*rep_baseline_forcing_term];
end
