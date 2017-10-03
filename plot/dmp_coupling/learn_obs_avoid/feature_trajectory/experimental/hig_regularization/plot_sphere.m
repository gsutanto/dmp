function [] = plot_sphere(R, Ox, Oy, Oz)
    [X, Y, Z]   = sphere;
    surf(Ox + R*X, Oy + R*Y, Oz + R*Z);
end