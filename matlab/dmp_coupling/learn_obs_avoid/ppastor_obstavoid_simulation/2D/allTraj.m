function allTraj(method)
    close all;
    clc;

    x = linspace(-0.7,0.7,15);

    for i = 1:size(x,2);
        disp(i)
        testMultObst([x(i) -1]',i,method);
    end
end