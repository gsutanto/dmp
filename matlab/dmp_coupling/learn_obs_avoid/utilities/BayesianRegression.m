function [m, retain_idx, t_fit_hist, m_hist, log10_a_hist] = BayesianRegression(varargin)
% Implementation of Bayesian Learning with single (scalar) alpha.
% Does automated regularization tuning using EM Algorithm.
% Implemented from "Pattern Recognition and Machine Learning" textbook
% by Christopher M. Bishop, page 166 - 169
% Author: Giovanni Sutanto
% Date  : March 12, 2016

phi                 = varargin{1};
t_target            = varargin{2};
num_iter            = varargin{3};
debug_interval      = varargin{4};
debug               = varargin{5};
if (nargin > 5)
    alpha_min_threshold = varargin{6};
else
    alpha_min_threshold = 0;
end
if (nargin > 6)
    max_abs_weight_threshold    = varargin{7};
else
    max_abs_weight_threshold    = inf;
end

% dispose zero features, if any
nonzero_features_idx= any(phi);
retain_idx          = find(nonzero_features_idx==1)';
phi_new             = phi;
phi_new             = phi_new(:,retain_idx);
phi                 = phi_new;

N                   = size(phi,1);
M                   = size(phi,2);

t_fit_hist          = zeros(N, (num_iter/debug_interval));
m_hist              = zeros(M, (num_iter/debug_interval));
log10_a_hist        = zeros(1, (num_iter/debug_interval));

m                   = zeros(M, 1);

alpha               = 1;
beta                = 1;

prev_loglik         = -inf;

if debug
    A               = alpha * eye(M) + beta * phi'*phi;
    E               = (beta/2) * (norm(t_target - phi*m)^2) + (alpha/2) * (m'*m);
    try
        U         	= chol(A);  % upper triangular matrix (result of Cholesky decomposition on A)
    catch ME
        disp('WARNING: Cholesky Decomposition has failed.');

        return;
    end

    loglik          = (M/2.0)*log(alpha) + (N/2.0)*log(beta) - E ...
                      - 0.5*sum(log(diag(U))) - (N/2.0)*log(2*pi);
    display(['start: loglik=', num2str(loglik)]);
end

for i=1:num_iter
    A               = (alpha * eye(M)) + (beta * phi'*phi);
    try
        U         	= chol(A);  % upper triangular matrix (result of Cholesky decomposition on A)
    catch ME
        disp('WARNING: Cholesky Decomposition has failed.');

        return;
    end
    inv_U           = inv(U);
    inv_A           = inv_U * inv_U';
  
    new_m           = beta * inv_A * phi' * t_target;
    
    if (max(max(abs(new_m))) < max_abs_weight_threshold)
        m           = new_m;
    else
        disp(['WARNING: new maximum absolute m value above maximum threshold at iter ', num2str(i)]);
        % no updating on m;
        return;
%         keyboard;
    end
    
    gamma           = M - alpha * trace(inv_A);
    new_alpha       = gamma/(m'*m);
    
    if (new_alpha > alpha_min_threshold)
        alpha       = new_alpha;
    else
        disp(['WARNING: new alpha value below minimum threshold at iter ', num2str(i)]);
        % no updating on alpha;
        return;
%         keyboard;
    end
  
    beta            = (N-gamma)/(norm(t_target-phi*m)^2);  
    
    if((i >= 1) && (mod(i,debug_interval) == 0))
        t_fit = phi*m;

        t_fit_hist(:,(i/debug_interval))            = t_fit;
        m_hist(:,(i/debug_interval))                = m;
        log10_a_hist(1,(i/debug_interval))          = log10(alpha);

        mse = mean( (t_fit - t_target).^2 );
        if debug
            A               = alpha * eye(M) + beta * phi'*phi;
            E               = (beta/2) * (norm(t_target - phi*m)^2) + (alpha/2) * (m'*m);
            try
                U         	= chol(A);  % upper triangular matrix (result of Cholesky decomposition on A)
            catch ME
                disp('WARNING: Cholesky Decomposition has failed.');

                return;
            end
            
            prev_loglik     = loglik;

            loglik          = (M/2.0)*log(alpha) + (N/2.0)*log(beta) - E ...
                              - 0.5*sum(log(diag(U))) - (N/2.0)*log(2*pi);
                          
            if(loglik < prev_loglik)
                disp(['WARNING: Log-likelihood is decreasing. Previously was ', num2str(prev_loglik),...
                      ', current likelihood is ', num2str(loglik)]);
            end
        else
            loglik          = 0;
        end
        display(['i: ', num2str(i), ', mse=', num2str(mse),...
                 ' min w :', num2str(min(m)), ', max w: ',...
                 num2str(max(m)), ', loglik=', num2str(loglik)]);
    end
end