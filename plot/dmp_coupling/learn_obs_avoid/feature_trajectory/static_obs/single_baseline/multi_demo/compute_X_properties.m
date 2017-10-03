X           = dlmread('X.txt');
XTX         = X.' * X;
reg         = [1e-3; 1e-2; 1e-1];

    fprintf('rank(X)            = %d\n', rank(X));
    fprintf('rank(XTX)          = %d\n', rank(XTX));
    fprintf('cond(XTX)          = %d\n', cond(XTX));
    fprintf('\n');
for i = 1:size(reg,1)
    XTX_plus_R                  = XTX + (reg(i,1) * eye(size(XTX)));
    fprintf('reg                = %d\n', reg(i,1));
    fprintf('rank(XTX_plus_R)   = %d\n', rank(XTX_plus_R));
    fprintf('cond(XTX_plus_R)   = %d\n', cond(XTX_plus_R));
    fprintf('\n');
end