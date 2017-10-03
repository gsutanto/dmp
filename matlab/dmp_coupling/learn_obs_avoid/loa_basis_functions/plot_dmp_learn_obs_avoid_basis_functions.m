% Author: Giovanni Sutanto
% Date  : January 08, 2016
close all;
clc;

theta_lo            = 0;
theta_hi            = pi;
theta_N_grid        = 100;

d_lo                = 0;
d_hi                = 5.0;
d_N_grid            = 100;

% beta_lo             = 3.0/pi;
% beta_hi             = 45.0/pi;
% beta_N_grid         = 20;

beta_lo             = (1.0/pi);
beta_hi             = (3.0/pi);
beta_N_grid         = 20;

k_lo                = 3.0;
k_hi                = 45.0;
k_N_grid            = 10;

theta               = linspace(theta_lo, theta_hi, theta_N_grid);
beta                = linspace(beta_lo, beta_hi, beta_N_grid);

phi                 = cell(0,0);
figure;
hold        on;
for i = 1:size(beta,2)
    phi{i}          = zeros(size(theta));
    for j = 1:size(theta,2)
        %phi{i}(1,j) = theta(1,j) * exp(-beta(1,i)*theta(1,j));
        phi{i}(1,j) = (pi - theta(1,j)) * theta(1,j) * exp(-beta(1,i)*theta(1,j));
    end
    phi{i}          = phi{i}/max(phi{i});
    p_phi{i}        = plot(theta, phi{i});
    p_phi_legend{i} = strcat(['beta = ', num2str(beta(1,i)*pi), ' / pi']);
end
title('Basis Functions of the Obstacle Avoidance Coupling Term for Different Beta Values');
xlabel('theta');
ylabel('phi');
legend([p_phi{:}], p_phi_legend{:});
hold        off;



d                           = linspace(d_lo, d_hi, d_N_grid);
k                           = linspace(k_lo, k_hi, k_N_grid);

d_effect                    = cell(0,0);
figure;
hold        on;
for i = 1:size(k,2)
    d_effect{i}             = zeros(size(d));
    for j = 1:size(d,2)
        d_effect{i}(1,j)    = exp(-k(1,i)*d(1,j)*d(1,j));
    end
    p_d_effect{i}           = plot(d, d_effect{i});
    p_d_effect_legend{i}    = strcat(['k = ', num2str(k(1,i))]);
end
title('Effect of d on the Basis Functions of the Obstacle Avoidance Coupling Term for Different k Values');
xlabel('d');
ylabel('d_effect');
legend([p_d_effect{:}], p_d_effect_legend{:});
hold        off;


theta_grid  = theta_lo:((theta_hi-theta_lo)/(theta_N_grid-1)):theta_hi;
d_grid      = d_lo:((d_hi-d_lo)/(d_N_grid-1)):d_hi;
[theta, d]  = meshgrid(theta_grid, d_grid);
beta        = 5.0/pi;
k           = 0.5;
%phi         = theta .* exp(-beta * theta) .* exp(-k * (d .^ 2));
phi         = (pi - theta) .* theta .* exp(-beta * theta) .* exp(-k * (d .^ 2));
figure;
hold        on;
p_phi_surf  = surf(theta_grid, d_grid, phi);
set(p_phi_surf,'LineStyle','none');
title('A Basis Function of the Obstacle Avoidance Coupling Term for Beta=5.0/pi and k=0.5');
xlabel('theta');
ylabel('d');
zlabel('phi');
hold        off;