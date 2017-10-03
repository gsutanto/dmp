%***************************************************************
% Function: vbls.m
% Date:     April 2004 
%           Revised on August 2007
% Author:   Jo-Anne Ting
% Description:
%   Implements the Variational Bayesian Least Squares algorithm 
%   as written in Ting et al. (2005). Uses a Bayesian framework 
%   to regularize the Ordinary Least Squares solution 
%   against overfitting. 
%
% Inputs: 
%
%  where N = number of data samples, 
%        d = dimensionality of input data
%  x                    : training input (Nxd matrix)
%  y                    : training output (Nx1 matrix)
%  options              : data structure for algorithm settings 
%  options.threshold    : threshold value to check for convergence of 
%                         lower bound to the incomplete log likelihood
%  options.noise        : initial value for output data noise variance
%  options.numIterations: maximum number of EM iterations desired
%
% Outputs:
%
%  A data structure "result" containing the final converged
%  values of:    
%  result.meanX         : mean of training input x
%  result.meanY         : mean of training input y
%  result.stdX          : std deviation of training input x
%  result.stdY          : std deviation of training input y       
%  result.alpha_mean    : mean of the precision variables
%  result.a_alpha       : parameter a of the Gamma distribution
%  result.b_alpha       : parameter b of the Gamma distribution
%  result.b_mean        : mean of the regression variables
%  result.b_var         : variance of the regression variables
%  result.psi_y         : noise variance of y
%  result.psi_z         : noise variance of z
%  result.numIter       : max number of EM iterations to iterate 
%  result.LL            : lower bound to incomplete log likelihood
%
%***************************************************************
function [result] = vbls(x, y, options)	

% Check that we have the inputs needed
%--------------------------------------
if nargin < 2
  error('Insufficient inputs to vbls function');
end

if (~exist('options')|isempty(options))
  % Default settings for EM algorithm
  options.threshold     = 1e-6;
  options.noise         = 1;
  options.numIterations = 10000;
end

[N,d] = size(x); 


% Pre-process data to be mean zero and var 1
%---------------------------------------------
result.meanX = mean(x,1);
result.meanY = mean(y);
result.stdX = std(x,1);
result.stdY = std(y);
x = x - repmat(result.meanX, N, 1);
y = (y - result.meanY)./result.stdY;
for i=1:d
 if result.stdX(i) ~= 0
   x(:,i) = x(:,i) ./ repmat(result.stdX(i)', N, 1);
 end
end

% Initialization of parameters  (No changes needed here) 
%--------------------------------------------------------
a_alpha_init = 1e-8;  				
b_alpha_init = 1e-8*ones(d,1);  	
a_alpha      = a_alpha_init;
b_alpha      = b_alpha_init;
alpha_mean   = a_alpha ./ b_alpha;	
b_mean       = zeros(d,1);		
b_var        = ones(d,1)./alpha_mean;  
psi_y        = options.noise;	
psi_z        = options.noise*ones(d,1);	 

loglik       = -1.e10;  % Initial log likelihood value
LL           = [];      % keeps track of log likelihood value
result.count = 0;


% EM Algorithm
%--------------
for numIter = 1:options.numIterations
   
   % E-step
   %--------
   % Compute mean (Nxd) and variance (dx1) of hidden variables, z
   s           = psi_y + sum(psi_z./alpha_mean);  
   z_var       = psi_z./alpha_mean - (psi_z./alpha_mean).^2/s; 
   z_mean      = x*diag(b_mean) + 1/s*(y - x*b_mean)*(psi_z./alpha_mean)';
   z2_mean     = z_mean.^2 + repmat(z_var', N, 1);

   % Useful temporary variables (both dx1 vectors) 
   sum_x2_psiz = sum(x.^2,1)' + psi_z;  
   sum_zx      = sum(z_mean.*x,1)';        
    
   % Compute mean and variance of regression variable (both dx1 vectors)
   b_var       = (psi_z./alpha_mean)./sum_x2_psiz;  
   b_mean      = sum_zx./sum_x2_psiz;
   b2_mean     = b_mean.^2 + b_var;

   % Calculate mean of precision variable (dx1 vector)
   a_alpha     = a_alpha_init + N/2;
   b_alpha     = b_alpha_init ...
                + 0.5*(sum(z2_mean,1)' - (sum_zx.^2)./sum_x2_psiz )./psi_z;
   alpha_mean  = a_alpha ./ b_alpha;

 
   % Monitor the lower bound of complete log likelihood
   %----------------------------------------------------
   % Monitor once every 100 iterations (to reduce number of computations)
   if mod(numIter, 100) == 1
      oldloglik = loglik;

      % Calculate term needed for log likelihood expression
      z_mean_1  = sum(psi_z./alpha_mean)/s*y ...
                   + (1 - 1/s*sum(psi_z./alpha_mean))*x*b_mean;

      loglik = - N/2*log(psi_y) - 0.5/psi_y*sum((y - z_mean_1).^2) ...
         - N/2*sum(log(psi_z)) ...
         - 0.5*sum( (sum( z2_mean - 2*(z_mean.*x).*repmat(b_mean', N, 1) ...
         + (x.^2).*repmat(b2_mean', N, 1), 1)') .* alpha_mean ./ psi_z ) ...
         - sum(alpha_mean .* (b_alpha + 0.5*b2_mean) ) ...
         + sum( (a_alpha + 0.5*(N+1) - 1) .* psi(a_alpha) ) ...
         - (0.5*(N+1) - 1)*sum(log(b_alpha)) ...
        ... % Entropy terms
         + 0.5*sum(alpha_mean.*(b_var+1)) ...
         - sum(log(b_alpha)) ...
         + 0.5*log(det(diag(z_var)));

      if (loglik < oldloglik)
         disp(sprintf('Likelihood VIOLATION iter %d : %g < %g', ...
         numIter, loglik, oldloglik))
       elseif (abs(loglik - oldloglik)/abs(loglik) < options.threshold | ~isfinite(loglik))
         disp(sprintf('Likelihood ratio tolerance reached: (%g)-(%g) < %g',...
         loglik, oldloglik, options.threshold))
         break;
       end
   end

   if mod(numIter, 100) == 1
     fprintf('Iter %4i: loglik = %g \n', numIter, loglik);
   end
   % Keep track of the lower bound to the log incomplete likelihood
   LL(numIter,:) = loglik;
   
   result.count = result.count + 1;
   
   % M-step
   %--------
   z_mean_1      = sum(psi_z./alpha_mean)/s*y ...
                    + (1 - 1/s*sum(psi_z./alpha_mean))*x*b_mean;
   one_z_var_one = sum(psi_z./alpha_mean)*(1 - 1/s*sum(psi_z./alpha_mean));
   psi_y         = sum((y - z_mean_1).^2)/N  + one_z_var_one;   
   psi_z         = sum( ((z_mean - x*diag(b_mean)).^2) ...
                   .* repmat(alpha_mean',N,1),1)'/N ...
                   + alpha_mean.*z_var ...
		   + psi_z./sum_x2_psiz .* sum(x.^2,1)'/N;
end 

% Fill the output data structure 
%--------------------------------
result.alpha_mean = alpha_mean;
result.a_alpha    = a_alpha;
result.b_alpha    = b_alpha;
result.b_mean     = b_mean;
result.b_var      = b_var;
result.z_var      = z_var;
result.z_mean     = z_mean;
result.psi_y      = psi_y;
result.psi_z      = psi_z;
result.numIter    = numIter;
result.LL         = LL;
