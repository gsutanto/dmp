function [ theta ] = compute_theta( o, y, yd )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    theta           = zeros(size(y, 1), 1);
    yo              = o - y;
    for i=1:size(y,1)
        if ((norm(yd(i,:)) == 0.0) || (norm(yo(i,:)) == 0))
            theta(i,1)  = 0.0;
        else
            theta(i,1)  = acos(dot(yo(i,:), yd(i,:))/(norm(yo(i,:))*norm(yd(i,:))));
        end
    end
end

