function allTraj(method)
    close all;
    clc;

    [x3,y3,z3]  = meshgrid(linspace(-1.0,1.0,3),...
                           linspace(-1.0,-1.0,1),...
                           linspace(-1.0,1.0,3));
    x           = reshape(x3,[],1);
    y           = reshape(y3,[],1);
    z           = reshape(z3,[],1);
    
    nobs        = 50;
    % o = [0 0 0; 0.4 0 0; -0.4 0 0; -0.35 1.5 0; 0.43 1.5 0; -0.8 0.8 0; 0.8 0.5 0;
    %      -1.0 2.0 0; 1.0 2.0 0]';
    % o = [0 0 0]';
    % o = [0 1.5 0]';
    % o = [0 1.5 0; -0.5 1.5 0; 0.5 1.5 0]';
    % o = [0 0 0;
    %      0.4 0 0.5;
    %      -0.4 0 -0.3;
    %      -0.35 -1.5 0.1;
    %      0.43 1.5 1.5;
    %      -0.8 0.8 -0.7;
    %      0.8 0.5 -1;
    %      -1.0 2.0 0.2;
    %      3.5 -2.0 -2.5;
    %      2.5 0.5 0.5;
    %      1.5 -2.5 1.7;
    %      1.25 0.25 2]';
%     ox  = -1 + 2*rand(1,nobs);
%     oy  = 0.5 + 2*rand(1,nobs);
%     oz  = -1 + 2*rand(1,nobs);
    ox  = -2 + 4*rand(1,nobs);
    oy  = -1.5 + 6*rand(1,nobs);
    oz  = -2 + 4*rand(1,nobs);
    o   = [ox;oy;oz];

    for i = 1:size(x,1);
        disp(i)
        testMultObst([x(i) y(i) z(i)]', i, o,method);
    end
end