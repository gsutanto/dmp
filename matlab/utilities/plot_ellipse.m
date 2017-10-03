function [ h ] = plot_ellipse( cx, cy, rx, ry )
    th      = 0:pi/50:2*pi;
    x_ell   = rx * cos(th) + cx;
    y_ell   = ry * sin(th) + cy;
    h       = plot(x_ell, y_ell);
end