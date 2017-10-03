% R=rotmatrix(axis,theta)

function R=rotmatrix(axis,theta)

x = axis(1);
y = axis(2);
z = axis(3);
c = cos(theta); s = sin(theta); ic = 1-c;
xs = x*s;   ys = y*s;   zs = z*s;
xC = x*ic;   yC = y*ic;   zC = z*ic;
xyC = x*yC; yzC = y*zC; zxC = z*xC;
R = [ x*xC+c   xyC-zs   zxC+ys; xyC+zs   y*yC+c   yzC-xs; zxC-ys   yzC+xs   z*zC+c ];


