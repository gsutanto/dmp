function [phi] = compute_features_lyapunov_basis_functions(Y,Yd,tau,be1,ko1,be2,ko3,obs3, object)

% Computing features for a ellipsoid - centre is in obs3 - average from
% Vicon markers
% Nearest points is the point on the surface, along the line joining the
% center and the end-effector

n1 = size(be1,1);
n2 = size(be2,2);
n3 = size(ko3,2);
the=0;
the_o=0;

r_obs_cyl = 0.03;
r_obs_sph = 0.05;
r_obs_ell = [0.09 0.045 0.045];

%radius of the ellipsoid - pre measured

r=[0.09 0.045 0.045];

computePhiDYN1  = 0;
computePhiDYN2  = 1;
computePhiG     = 1;
computePhiR     = 0;
computePhiS     = 0;

n       = 0;
if (computePhiDYN1)
    n   = n + n2;
end
if (computePhiDYN2)
    n   = n + n1;
end
if (computePhiG)
    n   = n + n3;
end
if (computePhiR)
    n   = n + n3;
end
if (computePhiS)
    n   = n + n3;
end

phi = zeros(size(Y,1),n,3);

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
    
    zero_threshold  = 1e-5;
    
    if ((norm(yd)*norm(pc))<zero_threshold)
        cos_thetaC  = 0.0;
    else
        cos_thetaC  = yd*pc.'/(norm(yd)*norm(pc));
    end
    
    if ((norm(yd)*norm(pp))<zero_threshold)
        cos_thetaP  = 0.0;
    else
        cos_thetaP  = yd*pp.'/(norm(yd)*norm(pp));
    end
    
    if (computePhiDYN1)
        % based on U_dyn:
        for i=1:size(be2,2)
            if ((norm(pc)<zero_threshold) || (cos_thetaC<=0.0))
                phiDYN1(:,i)=zeros(3,1);
            else
                phiDYN1(:,i)=tau*(cos_thetaC^(be2(1,i)-1))*(1/(norm(pc)^4))*((yd*pc.'*(-pc.')) + (be2(1,i)*[(-pc.'),yd.']*[yd*pc.';(norm(pc)^2)]));
            end
            if ((norm(pp)<zero_threshold) || (cos_thetaP<=0.0))
                phiDYN1(:,i)=phiDYN1(:,i);
            else
                phiDYN1(:,i)=phiDYN1(:,i)+tau*(cos_thetaP^(be2(1,i)-1))*(1/(norm(pp)^4))*((yd*pp.'*(-pp.')) + (be2(1,i)*[(-pp.'),yd.']*[yd*pp.';(norm(pp)^2)]));
            end
        end
    end
    

    if (computePhiG)
        phiG        = zeros(3,size(ko3,2));
    end
    if (computePhiR)
        phiR        = zeros(3,size(ko3,2));
    end
    if (computePhiS)
        phiS        = zeros(3,size(ko3,2));
    end
    for i=1:size(ko3,2)
        if (computePhiG)
            phiG(:,i)   = -2.0*tau*(1/ko3(1,i)^2)*exp(-(1/ko3(1,i)^2)*(pc*pc.'))*pc.';              % for obstacle center
            phiG(:,i)   = phiG(:,i) -2.0*tau*(1/ko3(1,i)^2)*exp(-(1/ko3(1,i)^2)*(pp*pp.'))*pp.';    % for obstacle at point p
        end
        if (computePhiR)
            phiR(:,i)   = -2.0*tau*((ko3(1,i)/(pc*pc.'))^2)*pc.';               % for obstacle center
            phiR(:,i)   = phiR(:,i) -2.0*tau*((ko3(1,i)/(pp*pp.'))^2)*pp.';     % for obstacle at point p
        end
        % phiQ cannot be used, because it is NOT positive semi-definite
%         phiQ(:,i)   = -2.0*pc.';
%         phiQ(:,i)   = phiQ(:,i) -2.0*pp.';
        if (computePhiS)
            phiS(:,i)   = tau*[(sqrt(pc*pc.')<=ko3(1,i))*(1.0/(sqrt(pc*pc.'))^3)*((1.0/sqrt(pc*pc.'))-(1.0/ko3(1,i)))*(-pc.')];
            phiS(:,i)   = phiS(:,i) + tau*[(sqrt(pp*pp.')<=ko3(1,i))*(1.0/(sqrt(pp*pp.'))^3)*((1.0/sqrt(pp*pp.'))-(1.0/ko3(1,i)))*(-pp.')];
        end
    end
    
    if (computePhiDYN2)
        % based on U_dyn2:
    %     fprintf('Evaluating phiDYN2\n');
        for i=1:size(be1,1)
            if ((norm(pc)<zero_threshold) || (norm(yd)<zero_threshold))
                phiDYN2(:,i)=zeros(3,1);
            else
                phiDYN2(:,i)=tau*[-norm(yd)*((2*ko1(i,1)*pc.')-((be1(i,1)/(norm(yd)*norm(pc)))*yd.')+((be1(i,1)/(norm(yd)*(norm(pc))^3))*(yd*pc.')*pc.'))*exp((((be1(i,1)/(norm(yd)*norm(pc)))*yd.')-(ko1(i,1)*(pc.'))).'*pc.')];
            end
            if ((norm(pp)<zero_threshold) || (norm(yd)<zero_threshold))
                phiDYN2(:,i)=phiDYN2(:,i);
            else
                phiDYN2(:,i)=phiDYN2(:,i)+tau*[-norm(yd)*((2*ko1(i,1)*pp.')-((be1(i,1)/(norm(yd)*norm(pp)))*yd.')+((be1(i,1)/(norm(yd)*(norm(pp))^3))*(yd*pp.')*pp.'))*exp((((be1(i,1)/(norm(yd)*norm(pp)))*yd.')-(ko1(i,1)*(pp.'))).'*pp.')];
            end
        end
    end
    
    %all the features are stacked together - in all the dimensions
    phitmp = [];
    if (computePhiDYN1)
        phitmp = [phitmp; phiDYN1.'];
    end
    if (computePhiDYN2)
        phitmp = [phitmp; phiDYN2.'];
    end
    if (computePhiG)
        phitmp = [phitmp; phiG.'];
    end
    if (computePhiR)
        phitmp = [phitmp; phiR.'];
    end
    if (computePhiS)
        phitmp = [phitmp; phiS.'];
    end
    phi(j,:,:) = phitmp;
end

end
