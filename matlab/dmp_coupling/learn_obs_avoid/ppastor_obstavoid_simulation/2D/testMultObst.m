function testMultObst(x,ind,method)

dt = 0.01;
t=0;
%x = [1 1]';
v = [0 0]';
%g = [-0.1 0.0]';
g = [0.01 3]';
K = [1 0 ; 0 1];
D = [2 0 ; 0 2];

% o = [0 0; 0.4 0; -0.4 0; -0.35 1.5; 0.43 1.5; -0.8 0.8; 0.8 0.5;
%      -1.0 2.0; 1.0 2.0]';
% o = [0 0]';
% o = [0 1.5]';
% o = [0 1.5; -0.5 1.5; 0.5 1.5]';
o = [0 0; 0.4 0; -0.4 0; -0.35 1.5; 0.43 1.5; -0.8 0.8; 0.8 0.5;
     -1.0 2.0; 3.5 2.0; 2.5 0.5; 1.5 2.5; 1.25 0.25]';

N = 1000;
tt = zeros(1,N);
xx = zeros(2,N);
vv = zeros(2,N);
EE = zeros(1,N);
x3 = zeros(3,1);
v3 = zeros(3,1);
ox3 = zeros(3,1);
numo = size(o,2);

gammaPP     = 6;
% betaPP      = 10/pi;  % default (from Peter Pastor's simulation)
betaPP      = 10/pi;
lambdaDYN1  = 1.0;
betaDYN1    = 2.0;
gammaA      = 6;
betaA       = 10/pi;
kA          = 3;
sigsq       = 0.3;
betaDYN2    = 1.0;
kDYN2       = 1.0;

for i=1:N
   
    v3(1:2) = v;
    dvobst=[0 0]';
    for j=1:numo
        ox3(1:2)        = o(:,j)-x;
        if (method==0)      % Peter Pastor's
            dvobst3     = computePastorICRA2009ObstAvoidCt( gammaPP, betaPP, ox3, v3 );
        elseif (method==1)  % Dae-Hyung Park's
            dvobst3     = computeParkHumanoids2008ObstAvoidCt( lambdaDYN1, betaDYN1, ox3, v3 );
        elseif (method==2)
            dvobst3     = computeAksharaHumanoids2014ObstAvoidCt( gammaA, betaA, kA, ox3, v3 );
        elseif (method==3)
            dvobst3     = computeLyapunovGaussianObstAvoidCt( sigsq, ox3 );
        elseif (method==4)
            dvobst3     = computeParkHumanoids2008ObstAvoidCt( lambdaDYN1, betaDYN1, ox3, v3 );
            dvobst3     = dvobst3 + computeLyapunovGaussianObstAvoidCt( sigsq, ox3 );
        elseif (method==5)
            dvobst3     = computeLyapunovDyn2ObstAvoidCt( betaDYN2, kDYN2, ox3, v3 );
        end
        dvobst          = dvobst + dvobst3(1:2);
    end
    %phi = atan2(v(2),v(1));
    %psi = atan2(-x(2),-x(1));
    %dphi = gamma*(phi-psi)*exp(-4*(phi-psi)^2);
    %dphi = gamma*(phi-psi)*exp(-10/pi*abs(phi-psi));
    %dvobst = [-v(2) v(1)]' * dphi;
        
    E=0.5*(g-x)'*K*(g-x) + 0.5*v'*v;    

    tt(i) = t;
    xx(:,i) = x;
    vv(:,i) = v;
    EE(i) = E;
    
    dx = v;
    dv = K*(g-x) - D*v + dvobst;
    
    x = x + dx*dt;
    v = v + dv*dt;
    t = t + dt;
end

%subplot(2,1,1)
plot(o(1,:),o(2,:),'ok'); hold on
plot(xx(1,:),xx(2,:),'b');
%subplot(2,1,2)
%plot(EE);

% traj = xx';
% fname = sprintf('traj_mult%d.dat',ind);
% save('-ASCII',fname,'traj');
