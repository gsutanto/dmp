function [traj_new] = completeTrajectory(varargin)
traj    = varargin{1};
dt      = varargin{2};
tau     = varargin{3};
if (nargin > 3)
    tau_multiplier  = varargin{4};
else
    tau_multiplier  = 1.1;
end

[T, dim] = size(traj);

%increase length by tau_multiplier factor
taunew = tau*tau_multiplier;
Tnew = round(taunew/dt);
add_points = Tnew - T;

if(mod(add_points,2) > 0 )
   add_points = add_points +1;
   Tnew = Tnew+1;
end

traj_new = zeros(Tnew, dim);
for d = 1:dim
    try
        traj_new(:,d) = [traj(1,d)*ones(add_points/2,1); traj(:,d); traj(end,d)*ones(add_points/2,1)];
    catch ME
        keyboard;
    end
end
% for d = 1:dim
%     td = diffnc(traj(:,d), dt);
%     
%     %spline interpolation to go finish velocity profile
%     x = [T-2, T-1, Tnew];
%     y = [td(T-2)-td(T-3) td(T-2) td(T-1) 0 0];
%     xx = (T-2):Tnew;
%     yy = spline(x, y, xx);
%     tdnew = [td(1:T-3); yy'];
%     tnew = cumsum(tdnew)*dt;
%     traj_new(:,d) = tnew;
% end

end