function [ r, norm_r, yo ] = compute_rotation_axes( o, y, yd )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    r               = zeros(size(y));
    norm_r          = zeros(size(y, 1), 1);
    yo              = o - y;
    for i=1:size(y,1)
        r(i,:)      = cross(yo(i,:), yd(i,:));
        norm_r(i,:) = norm(r(i,:));
    end
end

