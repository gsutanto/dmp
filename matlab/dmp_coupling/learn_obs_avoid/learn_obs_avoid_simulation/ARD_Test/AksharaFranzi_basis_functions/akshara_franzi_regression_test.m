clear all;
close all;
clc;

n_theta_grid    = 30;
n_d_grid        = 180;
[theta_mgrid, d_mgrid]  = meshgrid(linspace(0.0,pi,n_theta_grid),...
                                   linspace(0.0,29.0,n_d_grid));
theta_grid      = reshape(theta_mgrid, n_theta_grid*n_d_grid, 1);
d_grid          = reshape(d_mgrid, n_theta_grid*n_d_grid, 1);

n_beta_grid     = 5;
n_k_grid        = 5;
[beta_mgrid, k_mgrid]   = meshgrid(linspace(6.0/pi,14.0/pi,n_beta_grid),...
                                   linspace(10,30,n_k_grid));
beta_grid       = reshape(beta_mgrid, n_beta_grid*n_k_grid, 1);
k_grid          = reshape(k_mgrid, n_beta_grid*n_k_grid, 1);

n_basis         = n_beta_grid*n_k_grid;
Phi             = zeros(n_theta_grid*n_d_grid, n_basis);

for i=1:n_theta_grid*n_d_grid
    phi         = theta_grid(i,1) * exp(-beta_grid * theta_grid(i,1)) .* exp(-k_grid * (d_grid(i,1)^2));
    Phi(i,:)    = phi';
end

% rank(Phi)

w_SYNTH                         = zeros(n_basis,1);
w_SYNTH(round(n_basis/2),1)     = 3.5;
w_SYNTH(round(n_basis/4),1)     = 15.0;
w_SYNTH(round(3*n_basis/4),1)   = 7.0;

Ct          = Phi * w_SYNTH;

XX          = Phi.'*Phi;
xc          = Phi.'*Ct;
reg         = 1e-9;
A           = reg*eye(size(XX,2));
w           = (A + XX)\xc;

w_ard       = zeros(n_basis,1);
[w_ard_d,r_idx] = ARD(Phi,Ct, 0);
w_ard(r_idx,1)  = w_ard_d;

% figure;
% hold on;
% plot(x,y,'r');
% for i=1:n_basis
%     plot(x,phi(:,i),'b');
% end
% hold off;

figure;
hold on;
plot(w_SYNTH,'ro');
plot(w,'bx');
plot(w_ard,'g+');
legend('synthetic','regular ridge regression', 'ARD');
title('weights distribution')
hold off;
