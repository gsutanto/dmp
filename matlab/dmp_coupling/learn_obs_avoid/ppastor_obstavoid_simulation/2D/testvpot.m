function testvpot

K=1;
D=1;
g=-1;
gamma=0.5;
x=1;
v=0;
t=0;
dt=0.001;

maxn=20000;
xx=zeros(maxn,1);
vv=zeros(maxn,1);
tt=zeros(maxn,1);

for n = 1:maxn
    xx(n)=x;
    vv(n)=v;
    tt(n)=t;
    dv = (K*(g-x) - D*v - gamma*v/x^2)*dt;
    dx = v*dt;
    t = t+dt;
    x = x+dx;
    v = v+dv;
end

plot(tt,xx,'-r');
hold on 
plot(tt,vv,'-g');
hold off
    