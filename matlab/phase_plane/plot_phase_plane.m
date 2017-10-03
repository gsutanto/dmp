% Modified from:
% http://matlab.cheme.cmu.edu/2011/08/09/phase-portraits-of-a-system-of-odes/

clear all; close all; clc;

tau     = 1.0;
alpha_v = 1.0;
beta_v  = 1.0/4.0;
g       = 0.0;
k       = 0.3;
o       = 1.5;

% DMP:
% Trial Model 1:
% f = @(t,Y) [(Y(2)/tau)*(1-exp(-k*(Y(1)-o)^2));...
%             ((alpha_v/tau)*((beta_v*(g-Y(1)))-Y(2)))*(1-exp(-k*(Y(1)-o)^2))];
% Trial Model 2:
% f = @(t,Y) [(Y(2)/tau)*(1-(1-(Y(1)-o)^2)*exp(-k*(Y(1)-o)^2));...
%             ((alpha_v/tau)*((beta_v*(g-Y(1)))-Y(2)))*(1-(1-(Y(1)-o)^2)*exp(-k*(Y(1)-o)^2))];
f = @(t,Y) [(Y(2)/tau)*(1-(exp(-k*(Y(1)-o)^2))^2);...
            ((alpha_v/tau)*((beta_v*(g-Y(1)))-Y(2)))*(1-(exp(-k*(Y(1)-o)^2))^2)];
% ORIGINAL EXAMPLE:
% f = @(t,Y) [Y(2); -sin(Y(1))];
% EXAMPLES FROM AME552 CLASS (SPRING 2016), LECTURE 2:
% f = @(t,Y) [-Y(1); 1-(Y(1))^2-(Y(2))^2];
% f = @(t,Y) [Y(2)-Y(1); 2*(Y(2))-(Y(1))^2];
% f = @(t,Y) [16*Y(1)^2 + 16*Y(2)^2 - 25.0; 16*Y(1)^2 - 16*Y(2)^2];
y1 = linspace(-2.5,2.5,51);
y2 = linspace(-2.5,2.5,51);

% creates two matrices one for all the x-values on the grid, and one for
% all the y-values on the grid. Note that x and y are matrices of the same
% size and shape, in this case 20 rows and 20 columns
[x,y] = meshgrid(y1,y2);
size(x)
size(y)

u = zeros(size(x));
v = zeros(size(x));

% we can use a single loop over each element to compute the derivatives at
% each point (y1, y2)
t=0; % we want the derivatives at each point at t=0, i.e. the starting time
for i = 1:numel(x)
    Yprime = f(t,[x(i); y(i)]);
    u(i) = Yprime(1);
    v(i) = Yprime(2);
end

quiver(x,y,u,v,'r'); figure(gcf)
xlabel('y_1')
ylabel('y_2')
axis tight equal;

% hold on
% for y20 = [0 0.5 1 1.5 2 2.5]
%     [ts,ys] = ode45(f,[0,50],[0;y20]);
%     plot(ys(:,1),ys(:,2))
%     plot(ys(1,1),ys(1,2),'bo') % starting point
%     plot(ys(end,1),ys(end,2),'ks') % ending point
% end
% hold off

'done'


% categories: ODEs
% tags: math
% post_id = 799; %delete this line to force new post;
