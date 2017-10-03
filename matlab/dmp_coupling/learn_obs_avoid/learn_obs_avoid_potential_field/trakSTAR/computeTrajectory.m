function [Y, Yd, Ydd, F] = computeTrajectory(n_rfs,tau,dt,w,y0,g,len)

global dcps;

ID        = 1;

dcp_franzi('init',ID,n_rfs,'letter_dcp');
dcp_franzi('reset_state',ID, y0);
dcp_franzi('set_goal',ID,g,1);

dcps(ID).w=w;

Y = zeros(len,1);
Yd = Y; Ydd=Y;
F = Y;
for i=1:len
  [y,yd,ydd,f]=dcp_franzi('run',ID,tau,dt);
  
%   Z(i,:)   = [dcps(ID).z dcps(ID).zd];
  Y(i)   = y;
  Yd(i)   = yd;
  Ydd(i)   = ydd;
  F(i) = f;
  
%   X(i,:)   = [dcps(ID).x dcps(ID).xd];
%   V(i,:)   = [dcps(ID).v dcps(ID).vd];
%   PSI(i,:) = dcps(ID).psi';
%   W(i,:)   = dcps(ID).w';
end;


end