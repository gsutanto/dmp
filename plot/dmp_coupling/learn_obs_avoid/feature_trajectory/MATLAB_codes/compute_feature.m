function [ feature, R_vector ] = compute_feature( o, y, yd, r, theta, beta, k, tau )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    yo              = o - y;
    d               = zeros(size(y, 1), 1);
    R_vector        = zeros(3, 3, size(y, 1));
    feature         = zeros(size(y, 1), 3);
    for i=1:size(y,1)
        d(i, 1)         = norm(yo(i, :));
        R               = vrrotvec2mat([r(i, 1:3),(pi/2)].');
        R_vector(:,:,i) = R;
        feature(i,:)    = (tau * R * yd(i, :).' * theta(i, 1) * exp(-beta * theta(i, 1)) * exp(-k * (d(i, 1)^2))).';
    end
end

