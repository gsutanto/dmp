function [m,rind] = ARD(phi,ct, debug)

% abs_weights_threshold   = 90.0;
abs_weights_threshold   = 1e10;

N=size(phi,1);
M=size(phi,2);

% rind=1:M;
rind        = find(range(phi,1)~=0);
phi_new     = phi(:,rind);

new_rind    = rind';
rind        = new_rind;
phi         = phi_new;

a=ones(size(rind,1),1);
b=1;

if debug
    iA = diag(1.0./a);
    C = 1.0/b*eye(N) + phi*iA*phi';
    % keyboard;
    cL = chol(C);

    loglik = -0.5*(N*log(2*pi)+ 2*sum(log(diag(cL))) + ct'*(cL\(cL'\ct)));
    display(['start: loglik=', num2str(loglik)])
end

for i=1:400
  
  A=diag(a);
  
  % a bit more robust
  Si = b*(phi'*phi) + A;
  L = chol(Si);
  Li = inv(L);
  Si = Li*Li';
  
  new_m = b*Si*phi'*ct;
  
  if ((min(new_m) <= -abs_weights_threshold) || ...
      (max(new_m) >= abs_weights_threshold))
      display('WARNING: weights threshold is reached');
      return;
  end
  
  m     = new_m;
  rind  = new_rind;
  
  diagS = sum(Li.^2,2);
  gamma = 1 - a.*diagS;
  a= gamma./(m.*m);
  
  neg_idx = find(a<0);
  if(~isempty(neg_idx))
    display('WARNING: negative precision');
    return;
%     keyboard;
  end
  
  %     S=inv(A+b*(phi'*phi) );
  %     m = b*S*phi'*ct;
  %     lam= 1- a.*diag(S);%diag(eye(size(A))-A*diag(diag(S)));
  %     aprev = a;
  %     a=lam./(m.*m);
  
  b=(N-sum(gamma))/(norm(ct-phi*m).^2);
  
  if(i >= 1 && mod(i,1) == 0)
    cfit = phi*m;
    mse = mean( (cfit - ct).^2 );
    loglik = 0;
    if debug
        iA = diag(1.0./a);
        C = 1.0/b*eye(N) + phi*iA*phi';
        cL = chol(C);
        loglik = -0.5*(N*log(2*pi)+ 2*sum(log(diag(cL))) + ct'*(cL\(cL'\ct)));
    end
    display(['i: ', num2str(i), ', mse=', num2str(mse), ', M: ',...
        num2str(length(a)), ' min w :', num2str(min(m)), ', max w: ',...
        num2str(max(m)), ', loglik=', num2str(loglik)]);
    if (mse == 0)
        return;
    end
  end
  count=0;
  phi_new=phi;
  atemp=a;
  
  % pruning of some basis functions
  for j=1:length(a)
    if(a(j)>10^3)
      phi_new=[phi_new(:,1:j-1-count) phi_new(:,j+1-count:end)];
      atemp=[atemp(1:j-1-count); atemp(j+1-count:end)];
      new_rind=[new_rind(1:j-1-count); new_rind(j+1-count:end)];
      count=count+1;
    end
  end
  a=atemp;
  
  %this is a hack to prevent negative precisions
%   a(a < 1e-12) = 1e-12;
  phi=phi_new;
end

%recompute mean one more time in case some values of "a" were pruned out in
%the last iteration
A=diag(a);

Si = b*(phi'*phi) + A;
L = chol(Si);
Li = inv(L);

m = b*((Li*Li')*phi'*ct);
% keyboard;