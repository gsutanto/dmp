function [m, retain_idx, t_fit_hist, m_hist, log_a_hist] = ARD(phi, t_target, param_table, num_iter, debug_interval, debug)

% dispose zero features, if any
nonzero_features    = any(phi);
retain_idx          = find(nonzero_features==1)';
phi_new             = phi;
phi_new             = phi_new(:,retain_idx);
phi                 = phi_new;

N                   = size(phi,1);
M                   = size(phi,2);

a_min_threshold     = 1e-10;

t_fit_hist          = zeros(N, (num_iter/debug_interval));
m_hist              = zeros(M, (num_iter/debug_interval));
log_a_hist          = zeros(M, (num_iter/debug_interval));

a=ones(M,1);
b=1;

if debug
    iA = diag(1.0./a);
    C = 1.0/b*eye(N) + phi*iA*phi';
    % keyboard;
    cL = chol(C);

    loglik = -0.5*(N*log(2*pi)+ 2*sum(log(diag(cL))) + t_target'*(cL\(cL'\t_target)));
    display(['start: loglik=', num2str(loglik)])
end

for i=1:num_iter
  
  A=diag(a);
  
  % a bit more robust
  Si = b*(phi'*phi) + A;
  L = chol(Si);
  Li = inv(L);
  Si = Li*Li';
  
  m = b*Si*phi'*t_target;
  
  w_stacked             = [param_table(retain_idx,:),m];
  
  %diagS                 = sum(Li.^2,2);    % this is equivalent to diag(Si.*eye(size(Si,1)))
  diagS                 = diag(Si.*eye(size(Si,1)));
  gamma                 = 1 - a.*diagS;
  new_a                 = gamma./(m.*m);
  valid_new_a_idx       = find(new_a > a_min_threshold);
  
  if(length(valid_new_a_idx) ~= length(a))
    display(['WARNING: some alpha values below threshold at iter ', num2str(i)]);
%     keyboard;
  end
  
  a(valid_new_a_idx,:)  = new_a(valid_new_a_idx,:);
  
  %     S=inv(A+b*(phi'*phi) );
  %     m = b*S*phi'*t_target;
  %     lam= 1- a.*diag(S);%diag(eye(size(A))-A*diag(diag(S)));
  %     aprev = a;
  %     a=lam./(m.*m);
  
  b=(N-sum(gamma))/(norm(t_target-phi*m).^2);
  
    
  if(i > 1 && mod(i,debug_interval) == 0)
    t_fit = phi*m;
    
    t_fit_hist(:,(i/debug_interval))     = t_fit;
    m_hist(retain_idx,(i/debug_interval))     = m;
    log_a_hist(retain_idx,(i/debug_interval)) = log(a);
    
    mse = mean( (t_fit - t_target).^2 );
    loglik = 0;
    if debug
        iA = diag(1.0./a);
        C = 1.0/b*eye(N) + phi*iA*phi';
        cL = chol(C);
        loglik = -0.5*(N*log(2*pi)+ 2*sum(log(diag(cL))) + t_target'*(cL\(cL'\t_target)));
    end
    display(['i: ', num2str(i), ', mse=', num2str(mse), ', M: ',...
        num2str(length(a)), ' min w :', num2str(min(m)), ', max w: ',...
        num2str(max(m)), ', loglik=', num2str(loglik)]);
  end
  
  neg_idx = find(a<0);
  if(~isempty(neg_idx))
    display('WARNING: negative precision');
    keyboard;
  end
  count=0;
  phi_new=phi;
  atemp=a;
  
  % pruning of some basis functions
  for j=1:length(a)
    if(a(j)>10^3)
      phi_new=[phi_new(:,1:j-1-count) phi_new(:,j+1-count:end)];
      atemp=[atemp(1:j-1-count); atemp(j+1-count:end)];
      retain_idx=[retain_idx(1:j-1-count); retain_idx(j+1-count:end)];
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

m = b*((Li*Li')*phi'*t_target);
% keyboard;