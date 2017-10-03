function testPotEn

dt = 0.001;
t=0;
x = [1 1]';
v = [0 0]';
g = [-0.1 0.0]';
%g = [-1 -0.99]';
K = [1 0 ; 0 1];
D = [2 0 ; 0 2];
gamma = 0.5;

N = 80000;
tt = zeros(1,N);
xx = zeros(2,N);
vv = zeros(2,N);
EE = zeros(1,N);

for i=1:N
    
    costheta = -x'*v;
    if costheta<0
        costheta=0;
    end
    phi = costheta^2/((x'*x) * (v'*v)+1e-10) * norm(v)/(norm(x)+1e-10);
    E = 0.5*(g-x)'*K*(g-x) + 0.5*v'*v + phi;
    
    tt(i) = t;
    xx(:,i) = x;
    vv(:,i) = v;
    EE(i) = E;
    
    dx = v;
    dv = K*(g-x) - D*v - gamma*phi*(2*v/(v'*x+1e-10) - 3*x/(x'*x+1e-10)) ;
    
    x = x + dx*dt;
    v = v + dv*dt;
    t = t + dt;
end

%subplot(2,1,1)
plot(xx(1,:),xx(2,:));
%subplot(2,1,2)
%plot(EE);

%traj = xx';

%save -ASCII traj_old.dat traj
