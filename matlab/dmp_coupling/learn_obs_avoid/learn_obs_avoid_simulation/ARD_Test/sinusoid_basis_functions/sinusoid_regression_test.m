clear all;
close all;
clc;

x           = linspace(0,8.0*pi,500)';
n_basis     = 10;
f_basis     = linspace(0,9,n_basis);
w_SYNTH     = zeros(n_basis,1);
w_SYNTH(round(n_basis/2),1) = 2.0;
w_SYNTH(round(n_basis/4),1) = 15.0;
w_SYNTH(round(3*n_basis/4),1) = -7.0;
phi         = sin(x*f_basis);
y           = phi * w_SYNTH;

XX          = phi.'*phi;
xc          = phi.'*y;
reg         = 1e-9;
A           = reg*eye(size(XX,2));
w           = (A + XX)\xc;
w_ard       = zeros(n_basis,1);
[w_ard_d,r_idx] = ARD(phi,y, 0);
w_ard(r_idx,1)  = w_ard_d;

figure;
hold on;
plot(x,y,'r');
for i=1:n_basis
    plot(x,phi(:,i),'b');
end
hold off;

figure;
hold on;
plot(w_SYNTH,'ro');
plot(w,'bx');
plot(w_ard,'g+');
legend('synthetic','regular ridge regression', 'ARD');
title('weights distribution')
hold off;
