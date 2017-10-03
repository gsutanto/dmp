function [X] = diffnc(X,dt)
% [X] = diffc(X,dt) does non causal differentiation with time interval
% dt between data points. The returned vector (matrix) is of the same length
% as the original one

% 	Stefan Schaal December 29, 1995

[m,n] = size(X);
for i=1:n,
 XX(:,i) = conv(X(:,i),[1,0,-1]/2/dt);
end;

X = XX(2:m+1,:);
X(1,:) = X(2,:);
X(m,:)=X(m-1,:);