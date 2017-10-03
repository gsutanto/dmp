function V=vif(X)
% V=vif(X)
%
% Variance inflaction factor in Regression Analysis
% the variance inflation factor (VIF) quantifies the severity of 
% multicollinearity in an ordinary least squares regression analysis. 
% It provides an index that measures how much the variance of an estimated 
% regression coefficient is increased because of collinearity.
%
% INPUT:
% 
% X is the matrix n onservation x p variables  
%
% OUTPUT:
%
% V is a column vector of vif indices

[n,p]=size(X);
V=zeros(p,1);
if p>2
    for i=1:p
        pred=setdiff(1:p,i);
        [~, R2 , ~]=ridge_ols(X(:,i),X(:,pred),0);
        V(i)=1/(1-R2);
    end
else
    [~, R2 , ~]=ols(X(:,1),X(:,2),1);
    V(1)=1/(1-R2);
    V(2)=V(1);
end


