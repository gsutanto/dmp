function [w,F,Ft,c,D] = computeForcingTermMulti(type,Ts,n_rfs,dt)

global dcps;
nL = length(Ts);
ID        = 1;

dcp_franzi('init',ID,n_rfs,'letter_dcp');

dts = zeros(1,nL);
taus = zeros(1,nL);
goals = zeros(1,nL);
y0s = zeros(1,nL);
ns = zeros(1,nL);
T = cell(0,0);
Td = cell(0,0);
Tdd = cell(0,0);
for i=1:nL
    t = Ts{i};
    ns(i)         = length(t);
    dts(i)        = dt;%tau/ns(i);
    taus(i) = ns(i)*dts(i);
    goals(i) = t(end);
    y0s(i) = t(1);
    T{i} = t;
    Td{i} = diffnc(t,dts(i));
    Tdd{i} = diffnc(Td{i},dts(i));

end

[w,F,Ft,c,D] = dcp_franzi(type,ID,taus,dts,T,Td,Tdd);

end