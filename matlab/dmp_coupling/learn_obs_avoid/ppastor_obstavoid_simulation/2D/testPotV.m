function testPotV(x)

dt = 0.001;
t=0;
%x = [1 1]';
v = [0 0]';
%g = [-0.1 0.0]';
g = [0.01 0.5]';
K = [1 0 ; 0 1];
D = [2 0 ; 0 2];
gamma = 7;

N = 11000;
tt = zeros(1,N);
xx = zeros(2,N);
vv = zeros(2,N);
EE = zeros(1,N);
x3 = zeros(3,1);
v3 = zeros(3,1);

for i=1:N
    
    % we compute such that it also hold for more general 3D case
    x3(1:2) = x;
    v3(1:2) = v;
    
    x3 = x3/norm(x3);
    vl = norm(v3);
    if vl>0
        v3n = v3/vl;
    else
        v3n = v3;
    end
    rotaxis = cross(v3n,-x3);
    if vl>0
        rotaxis = rotaxis/norm(rotaxis);
    end
    R = rotmatrix(rotaxis,-pi/2);
    
    phi = acos(-v3n'*x3);
    
    dphi = gamma*phi*exp(-10/pi*phi);
   
    dvobst3 = dphi * R * v3;
    dvobst = dvobst3(1:2);
    
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
o = zeros(2,5);
plot(o(1,:),o(2,:),'ok'); hold on
plot(xx(1,:),xx(2,:),'b');
%subplot(2,1,2)
%plot(EE);

%traj = xx' + ones(size(xx,2),1)*[0.5 0.5];

%save -ASCII traj_new3.dat traj
