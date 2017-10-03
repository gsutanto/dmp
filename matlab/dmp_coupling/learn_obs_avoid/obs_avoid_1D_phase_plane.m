close   all;
clear   all;
clc;

th              = [0:0.2:pi];

beta            = 1.25;

thd_m0          = th .* exp(-beta .* th);
thd_m1          = th .* (pi - th) .* exp(-beta .* th);
thd_m2          = th .* (th - pi) .* exp(-beta .* th);

figure;
hold            on;
q0              = quiver(th, zeros(size(th)), thd_m0/(5*pi), zeros(size(th)));
% set(q0, 'AutoScale', 'off');
title('model: thd  = th .* exp(-beta .* th)');
hold            off;

figure;
hold            on;
q1              = quiver(th, zeros(size(th)), thd_m1/(5*pi), zeros(size(th)));
% set(q1, 'AutoScale', 'off');
title('model: thd  = th .* (pi - th) .* exp(-beta .* th)');
hold            off;

figure;
hold            on;
q2              = quiver(th, zeros(size(th)), thd_m2/(5*pi), zeros(size(th)));
% set(q2, 'AutoScale', 'off');
title('model: thd  = th .* (th - pi) .* exp(-beta .* th)');
hold            off;

th              = [0:0.001:pi];

thd_m0          = th .* exp(-beta .* th);
thd_m1          = th .* (pi - th) .* exp(-beta .* th);

figure;
hold            on;
p0              = plot(th, thd_m0);
p1              = plot(th, thd_m1);
title('Comparison between Two Obstacle Avoidance Models');
legend('model: thd  = th .* exp(-beta .* th)', 'model: thd  = th .* (pi - th) .* exp(-beta .* th)');
hold            off;

figure;
hold            on;
p0              = plot(th, thd_m0/max(thd_m0));
p1              = plot(th, thd_m1/max(thd_m1));
title('Normalized Comparison between Two Obstacle Avoidance Models');
legend('model: thd  = th .* exp(-beta .* th)', 'model: thd  = th .* (pi - th) .* exp(-beta .* th)');
hold            off;