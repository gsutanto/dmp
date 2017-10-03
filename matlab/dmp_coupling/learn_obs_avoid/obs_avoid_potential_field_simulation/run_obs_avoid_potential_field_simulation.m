close   all;
clear   all;
clc;

is_verifying_analytical_gradient_w_symbolic_math    = 0;

lambda  = 0.25;
beta    = 4.0;%1.0;     % It seems that the Lyapunov function by Park et al. is only positive definite for even-valued beta (odd-valued might be not).
sigsq   = 9.0;
obs_p   = [2.5; 2.5];   % obstacle coordinate
goal_p  = [0.0; 0.0];
ro      = 1.0;
po      = 1.0;          % parameter for Ustatic
beta2   = 1.0;          % parameter for Ugs
sigsq2  = 4.0;          % parameter for Ugs
muctheta    = 1.0;      % parameter for Udyn3
sigctheta   = 1.0;      % parameter for Udyn3

y1      = linspace(0.0, 5.0, 251);
y2      = linspace(0.0, 5.0, 251);

f_vel           = @(t,Y) [(goal_p-Y)/((norm(goal_p-Y)*(norm(goal_p-Y)~=0))+((norm(goal_p-Y)==0)))];
% f_vel           = @(t,Y) [-1.0;-1.0];
f_p_to_obs      = @(t,Y) [(obs_p-Y)];
f_cos_theta     = @(t,VEL,P_X) [VEL.'*P_X/(((norm(VEL)*norm(P_X))*(norm(VEL)*norm(P_X)>0.00000001))+(norm(VEL)*norm(P_X)<=0.00000001))];
f_U_dyn         = @(t,COSTHETA,VEL,P_X) [((COSTHETA>0)*(norm(P_X)>0)*lambda*(COSTHETA^beta)*norm(VEL)/((norm(P_X)*(norm(P_X)>0.00000001))+(norm(P_X)<=0.00000001)))];
f_U_gaussian    = @(t,P_X) [exp(-sigsq*P_X.'*P_X)];
f_phi_gaussian  = @(t,P_X) [-2*sigsq*exp(-sigsq*P_X.'*P_X)*P_X];
% f_U_radial      = @(t,P_X) [ro^2/(P_X.'*P_X)];
% f_phi_radial    = @(t,P_X) [-2*((ro/(P_X.'*P_X))^2)*P_X];
f_U_static      = @(t,P_X) [(sqrt(P_X.'*P_X)<=po)*0.5*((1.0/sqrt(P_X.'*P_X))-(1.0/po))^2];
f_phi_static    = @(t,P_X) [(sqrt(P_X.'*P_X)<=po)*(1.0/(sqrt(P_X.'*P_X))^3)*((1.0/sqrt(P_X.'*P_X))-(1.0/po))*(-P_X)];
f_U_gs          = @(t,VEL,P_X) [norm(VEL)*exp(((beta2*VEL)-(sigsq2*(P_X))).'*P_X)];
f_phi_gs        = @(t,VEL,P_X) [-((2*sigsq2*P_X)-(beta2*VEL))*norm(VEL)*exp(((beta2*VEL)-(sigsq2*(P_X))).'*P_X)];
f_U_dyn2        = @(t,VEL,P_X) [norm(VEL)*exp((((beta2/((1e-10 + norm(VEL))*(1e-10 + norm(P_X))))*VEL)-(sigsq2*(P_X))).'*P_X)];
f_phi_dyn2      = @(t,VEL,P_X) [-norm(VEL)*(norm(P_X)~=0)*((2*sigsq2*P_X)-((beta2/((1e-10 + norm(VEL))*(1e-10 + norm(P_X))))*VEL)+((beta2/((1e-10 + norm(VEL))*((norm(P_X))^3 + 1e-10)))*(VEL.'*P_X)*P_X))*exp((((beta2/((1e-10 + norm(VEL))*(1e-10 + norm(P_X))))*VEL)-(sigsq2*(P_X))).'*P_X)];
f_U_dyn3        = @(t,VEL,P_X) [norm(VEL)*exp(-0.5 * (((VEL.'*P_X)/((1e-10 + norm(VEL))*(norm(P_X) + 1e-10)))-muctheta)^2 / sigctheta^2) * exp(-0.5*sigsq2*P_X.'*P_X)];
f_phi_dyn3      = @(t,VEL,P_X) [-norm(VEL)*(norm(P_X)~=0)*...
                                ((sigsq2*P_X) - ...
                                 (((((VEL.'*P_X)/((1e-10 + norm(VEL))*(norm(P_X) + 1e-10)))-muctheta)/(sigctheta^2))*...
                                  ((((VEL.'*P_X)/((1e-10 + norm(VEL))*((norm(P_X))^3 + 1e-10)))*P_X)-((1.0/((1e-10 + norm(VEL))*(1e-10 + norm(P_X))))*VEL))))*...
                                exp(-0.5 * (((VEL.'*P_X)/((1e-10 + norm(VEL))*(norm(P_X) + 1e-10)))-muctheta)^2 / sigctheta^2) * exp(-0.5*sigsq2*P_X.'*P_X)];

% creates two matrices one for all the x-values on the grid, and one for
% all the y-values on the grid. Note that x and y are matrices of the same
% size and shape
[x,y]   = meshgrid(y1,y2);

% we can use a single loop over each element to compute the derivatives at
% each point (y1, y2)
t=0; % we want the derivatives at each point at t=0, i.e. the starting time
for i = 1:size(x,1)
    for j = 1:size(x,2)
        vel     = f_vel(t,[x(i,j); y(i,j)]);
        V(i,j,1)= vel(1);
        V(i,j,2)= vel(2);
        
        px          = f_p_to_obs(t,[x(i,j); y(i,j)]);
        PX(i,j,1)   = px(1);
        PX(i,j,2)   = px(2);
        
%         nPX(i,j)    = norm(px);
        
        costheta        = f_cos_theta(t,vel,px);
        cos_theta(i,j)  = costheta;
        U_dyn(i,j)      = f_U_dyn(t,costheta,vel,px);
        if (norm(obs_p-[x(i,j); y(i,j)])<0.00000001)
             U_dyn(i,j) = max(max(U_dyn));
        end
        
        U_gaussian(i,j) = f_U_gaussian(t,px);
        
        U_sum(i,j)      = U_dyn(i,j) + U_gaussian(i,j);
        
        phi_gaussian        = f_phi_gaussian(t,px);
        PHI_GAUSSIAN(i,j,1) = phi_gaussian(1);
        PHI_GAUSSIAN(i,j,2) = phi_gaussian(2);
        
%         U_radial(i,j)   = f_U_radial(t,px);
%         phi_radial          = f_phi_radial(t,px);
%         PHI_RADIAL(i,j,1)   = phi_radial(1);
%         PHI_RADIAL(i,j,2)   = phi_radial(2);
    
%         U_static(i,j)       = f_U_static(t,px);
%         phi_static          = f_phi_static(t,px);
%         PHI_STATIC(i,j,1)   = phi_static(1);
%         PHI_STATIC(i,j,2)   = phi_static(2);
        
        U_gs(i,j)           = f_U_gs(t,vel,px);
        phi_gs              = f_phi_gs(t,vel,px);
        PHI_GS(i,j,1)       = phi_gs(1);
        PHI_GS(i,j,2)       = phi_gs(2);
        
        U_dyn2(i,j)          = f_U_dyn2(t,vel,px);
        phi_dyn2             = f_phi_dyn2(t,vel,px);
        PHI_DYN2(i,j,1)      = phi_dyn2(1);
        PHI_DYN2(i,j,2)      = phi_dyn2(2);
        
        U_dyn3(i,j)          = f_U_dyn3(t,vel,px);
        phi_dyn3             = f_phi_dyn3(t,vel,px);
        PHI_DYN3(i,j,1)      = phi_dyn3(1);
        PHI_DYN3(i,j,2)      = phi_dyn3(2);
    end
end

% figure;
% hold on;
% quiver(x,y,V(:,:,1),V(:,:,2),'r');
% xlabel('x')
% ylabel('y')
% axis tight equal;
% hold off;
% 
% figure;
% hold on;
% quiver(x,y,PX(:,:,1),PX(:,:,2),'r');
% xlabel('x')
% ylabel('y')
% axis tight equal;
% hold off;

% figure;
% hold on;
% h1  = surf(x,y,cos_theta);
% set(h1,'LineStyle','none');
% xlabel('x')
% ylabel('y')
% zlabel('cos theta')
% axis tight equal;
% hold off;

figure;
hold on;
h2  = surf(x,y,U_dyn);
set(h2,'LineStyle','none');
title('U\_dyn Surface Plot');
xlabel('x')
ylabel('y')
zlabel('U\_dyn')
axis tight equal;
hold off;

if (is_verifying_analytical_gradient_w_symbolic_math)
    syms x1_sym x2_sym x_sym v1_sym v2_sym v_sym o1_sym o2_sym o_sym d_sym lambda_sym beta_sym sigctheta_sym sigsq2_sym cos_theta_sym theta_sym phi_dyn_sym phi_dyn_byhand U_dyn3_sym phi_dyn3_sym phi_dyn3_byhand model_diff;
    x_sym           = [x1_sym; x2_sym];
    v_sym           = [v1_sym; v2_sym];
    o_sym           = [o1_sym; o2_sym];

    cos_theta_sym   = v_sym.'*(o_sym-x_sym)/(norm(v_sym)*norm(o_sym-x_sym));
    theta_sym       = acos(cos_theta_sym);
    U_dyn_sym       = lambda_sym*((cos_theta_sym)^beta_sym)*norm(v_sym)/norm(o_sym-x_sym);
    phi_dyn_sym     = -gradient(U_dyn_sym, [x1_sym, x2_sym]);
    d_sym           = norm(o_sym-x_sym);
    U_dyn3_sym      = norm(v_sym) * exp(-0.5*((cos_theta_sym-muctheta)/sigctheta_sym)^2) * exp(-0.5*sigsq2_sym * d_sym^2);
    phi_dyn3_sym    = -gradient(U_dyn3_sym, [x1_sym, x2_sym]);
    phi_dyn3_byhand = -norm(v_sym) * (sigsq2_sym*(o_sym-x_sym) - (((cos_theta_sym-muctheta)/(sigctheta_sym^2))*((v_sym.'*(o_sym-x_sym))/(norm(v_sym)*norm(o_sym-x_sym)^3)*(o_sym-x_sym) - (1.0/(norm(v_sym)*norm(o_sym-x_sym))*v_sym)))) * exp(-0.5*((cos_theta_sym-muctheta)/sigctheta_sym)^2) * exp(-0.5*sigsq2_sym * d_sym^2);

    phi_dyn_byhand  = lambda_sym*((cos_theta_sym)^(beta_sym-1))*(1.0/(norm(o_sym-x_sym)^4))*((v_sym.'*(o_sym-x_sym)*(x_sym-o_sym))+(beta_sym*[(x_sym-o_sym),v_sym]*[v_sym.'*(o_sym-x_sym); (norm(o_sym-x_sym)^2)]));

    x1 = [1,3,8,39,10];
    x2 = [2,5,4,3,90];
    v1 = [2,49,90,84,75];
    v2 = [14,24,92,90,102];
    o1 = [0.5,2,9,6,3];
    o2 = [10.5,22,39,96,63];
    sigctheta_sym=sigctheta;
    sigsq2_sym = sigsq2;
    model_diff = 0.0;
    for i=1:size(x1,2)
        v1_sym  = v1(1,i);
        v2_sym  = v2(1,i);

        x1_sym  = x1(1,i);
        x2_sym  = x2(1,i);

        o1_sym  = o1(1,i);
        o2_sym  = o2(1,i);

        subs_sym    = simplify(subs(phi_dyn3_sym))
        subs_byhand = simplify(subs(phi_dyn3_byhand))
        diff        = norm(simplify(subs_sym - subs_byhand))
        model_diff  = model_diff + diff
    end
    keyboard
end

o1_sym          = obs_p(1,1);
o2_sym          = obs_p(2,1);
o_sym           = [o1_sym; o2_sym];
lambda_sym      = lambda;
beta_sym        = beta;
model_diff      = 0.0;
max_norm_phi_dyn    = 0.0;
for i = 1:size(x,1)
    for j = 1:size(x,2)
        v1_sym  = V(i,j,1);
        v2_sym  = V(i,j,2);
        
        x1_sym  = x(i,j);
        x2_sym  = y(i,j);
        
%         model_diff  = model_diff + norm(subs(phi_dyn_sym) - subs(phi_dyn_byhand))
        if ((norm([V(i,j,1); V(i,j,2)])>=0.001) || (norm(obs_p-[x(i,j); y(i,j)])>=0.1))
%             phi_dyn     = subs(phi_dyn_byhand);
            x_sym           = [x1_sym; x2_sym];
            v_sym           = [v1_sym; v2_sym];

            cos_theta_sym   = v_sym.'*(o_sym-x_sym)/(norm(v_sym)*norm(o_sym-x_sym));
            if (cos_theta_sym>0.0)
                phi_dyn     = lambda_sym*((cos_theta_sym)^(beta_sym-1))*(1.0/(norm(o_sym-x_sym)^4))*((v_sym.'*(o_sym-x_sym)*(x_sym-o_sym))+(beta_sym*[(x_sym-o_sym),v_sym]*[v_sym.'*(o_sym-x_sym); (norm(o_sym-x_sym)^2)]));
            else
                phi_dyn     = zeros(2,1);
            end
        else
            phi_dyn         = zeros(2,1);
        end
        PHI_DYN(i,j,1)      = phi_dyn(1);
        PHI_DYN(i,j,2)      = phi_dyn(2);
        
%         if (norm(phi_dyn) > max_norm_phi_dyn)
%             max_norm_phi_dyn= norm(phi_dyn);
%         end
    end
end

% PHI_DYN     = PHI_DYN/max_norm_phi_dyn;
PHI_SUM     = PHI_DYN + PHI_GAUSSIAN;

% figure;
% hold on;
% surf(x,y,nPX);
% % quiver(x,y,PX(:,:,1),PX(:,:,2),'r');
% title('PX Contour Plot');
% xlabel('x')
% ylabel('y')
% axis tight equal;
% hold off;

figure;
hold on;
quiver(x,y,PHI_DYN(:,:,1),PHI_DYN(:,:,2),'r');
contour(x,y,U_dyn, 20);
title('PHI\_DYN Contour Plot');
xlabel('x')
ylabel('y')
axis tight equal;
hold off;

figure;
hold on;
h3  = surf(x,y,U_gaussian);
set(h3,'LineStyle','none');
title('U\_gaussian Surface Plot');
xlabel('x')
ylabel('y')
zlabel('U\_gaussian')
axis tight equal;
hold off;

figure;
hold on;
quiver(x,y,PHI_GAUSSIAN(:,:,1),PHI_GAUSSIAN(:,:,2),'r');
contour(x,y,U_gaussian, 20);
title('PHI\_GAUSSIAN Contour Plot');
xlabel('x')
ylabel('y')
axis tight equal;
hold off;

% figure;
% hold on;
% h3  = surf(x,y,U_radial);
% set(h3,'LineStyle','none');
% title('U\_radial Surface Plot');
% xlabel('x')
% ylabel('y')
% zlabel('U\_radial')
% axis tight equal;
% hold off;

% figure;
% hold on;
% quiver(x,y,PHI_RADIAL(:,:,1),PHI_RADIAL(:,:,2),'r');
% contour(x,y,U_radial, 20);
% title('PHI\_RADIAL Contour Plot');
% xlabel('x')
% ylabel('y')
% axis tight equal;
% hold off;

% figure;
% hold on;
% h4  = surf(x,y,U_static);
% set(h4,'LineStyle','none');
% title('U\_static Surface Plot');
% xlabel('x')
% ylabel('y')
% zlabel('U\_static')
% axis tight equal;
% hold off;
% 
% figure;
% hold on;
% quiver(x,y,PHI_STATIC(:,:,1),PHI_STATIC(:,:,2),'r');
% contour(x,y,U_static, 20);
% title('PHI\_STATIC Contour Plot');
% xlabel('x')
% ylabel('y')
% axis tight equal;
% hold off;

% figure;
% hold on;
% h5  = surf(x,y,U_gs);
% set(h5,'LineStyle','none');
% title('U\_gs Surface Plot');
% xlabel('x')
% ylabel('y')
% zlabel('U\_gs')
% axis tight equal;
% hold off;
% 
% figure;
% hold on;
% quiver(x,y,PHI_GS(:,:,1),PHI_GS(:,:,2),'r');
% contour(x,y,U_gs, 20);
% title('PHI\_GS Contour Plot');
% xlabel('x')
% ylabel('y')
% axis tight equal;
% hold off;

figure;
hold on;
h6  = surf(x,y,U_dyn2);
set(h6,'LineStyle','none');
title('U\_dyn2 Surface Plot');
xlabel('x')
ylabel('y')
zlabel('U\_dyn2')
axis tight equal;
hold off;

figure;
hold on;
quiver(x,y,PHI_DYN2(:,:,1),PHI_DYN2(:,:,2),'r');
contour(x,y,U_dyn2, 20);
title('PHI\_DYN2 Contour Plot');
xlabel('x')
ylabel('y')
axis tight equal;
hold off;

figure;
hold on;
h7  = surf(x,y,U_dyn3);
set(h7,'LineStyle','none');
title('U\_dyn3 Surface Plot');
xlabel('x')
ylabel('y')
zlabel('U\_dyn3')
axis tight equal;
hold off;

figure;
hold on;
quiver(x,y,PHI_DYN3(:,:,1),PHI_DYN3(:,:,2),'r');
contour(x,y,U_dyn3, 20);
title('PHI\_DYN3 Contour Plot');
xlabel('x')
ylabel('y')
axis tight equal;
hold off;

% figure;
% hold on;
% h6  = surf(x,y,U_sum);
% set(h6,'LineStyle','none');
% title('U\_sum Surface Plot');
% xlabel('x')
% ylabel('y')
% zlabel('U\_sum')
% axis tight equal;
% hold off;
% 
% figure;
% hold on;
% quiver(x,y,PHI_SUM(:,:,1),PHI_SUM(:,:,2),'r');
% contour(x,y,U_sum, 20);
% title('PHI\_SUM Contour Plot');
% xlabel('x')
% ylabel('y')
% axis tight equal;
% hold off;