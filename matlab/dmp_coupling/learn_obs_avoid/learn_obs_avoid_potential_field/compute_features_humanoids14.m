function [phi] = compute_features_humanoids14(Y,Yd,tau,be1,ko1,ko3,obs3, object)

% Computing features for a ellipsoid - centre is in obs3 - average from
% Vicon markers
% Nearest points is the point on the surface, along the line joining the
% center and the end-effector

n1 = size(be1,1);
n2 = size(ko1,1);
n3 = size(ko3,2);
the=0;
the_o=0;

%radius of the objects - pre measured
%franzi: Giovanni - do these measures correspond to yours?
r_obs_cyl = 0.03;
r_obs_sph = 0.05;
r_obs_ell = [0.09 0.045 0.045];


phi = zeros(size(Y,1),n1+n2+n3,3);
% phipure = zeros(size(phi,1), size(phi,2));
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

    if(norm(obs3-y)*norm(yd)~=0 && (sum((obs3-y).*yd)/(norm(obs3-y)*norm(yd)))>=-1 && (sum((obs3-y).*yd)/(norm(obs3-y)*norm(yd)))<=1)
        %theta for abstacle center
        the=acos(sum((obs3-y).*yd)/(norm(obs3-y)*norm(yd)));
        % theta for point P
        the_o=acos(sum((op-y).*yd)/(norm(op-y)*norm(yd)));
    else
        % franzi: what happens if the 'if-statement' doesn't evaluatue to
        % true? 
        display('warning - theta computation')
    end
    % r = (o-y) X ydot
    rot_a=cross((obs3-y),yd);
    % R is rotation matrix with axis = r, theta = pi/2
    Rot_mat=vrrotvec2mat([rot_a pi/2]);
    
    % franzi: norm(y1 -y2) is not equal (y1 - y2)^2 which is what is used
    % in a Gaussian/rbf kernel
    % phi1, calculated on the grid
%     phi1t=the.*exp(-the.*be1).*exp(-norm((obs3-y)).*ko1);
%     phi1t=the.*exp(-the.*be1).*exp(-sum((obs3-y).^2).*ko1);
    % GSUTANTO's modification:
    phi1t=exp(-the^2.*be1).*exp(-sum((obs3-y).^2).*ko1);
%     phipure(j,1:25) = phi1t;
    phi1t=tau*the.*(pi-the).*phi1t;
    %phi2
%     phi2t=the_o.*exp(-the_o.*be1).*exp(-ko1.*norm((op_r)));
%     phi2t=the_o.*exp(-the_o.*be1).*exp(-ko1.*sum((op-y).^2));
    % GSUTANTO's modification:
    phi2t=exp(-the_o^2.*be1).*exp(-ko1.*sum((op-y).^2));
%     phipure(j,26:50) = phi2t;
    phi2t=tau*the_o.*(pi-the_o).*phi2t;
    %phi3
%     phi3t=exp(-ko3*norm((obs3-y)))';
%     phi3t=exp(-ko3*sum((obs3-y).^2))';
    % GSUTANTO's modification:
    phi3t=exp(-ko3*sum((obs3-y).^2))';
%     phipure(j, 51) = phi3t;
    phi3t=tau*phi3t;
    % features are actually R*yd*phi
    t=Rot_mat*yd';
    
    %all the features are stacked together - in all the dimensions
    phitmp = [(phi1t*t');phi2t*t';phi3t*t'];
    phi(j,:,:) = phitmp;
end

end