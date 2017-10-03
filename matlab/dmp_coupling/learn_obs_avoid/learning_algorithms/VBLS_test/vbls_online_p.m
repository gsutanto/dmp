%***************************************************************
% Function: vbls_online_p.m
% Date:     April 2004 
%           Revised August 2007
% Author:   Jo-Anne Ting
% Description:
%   Implements an incremental version of Variational Bayesian 
%   Least Squares algorithm as written in Ting et al. (2005). 
%   Uses a Bayesian framework to regularize the Ordinary Least 
%   Squares solution against overfitting. 
%
%   ** Assumes training data has been normalized in batch form
%
% Inputs: 
%
%  where N = number of data samples, 
%        d = dimensionality of input data
%  x                    : training input (Nxd matrix)
%  y                    : training output (Nx1 matrix)
%  options              : data structure for algorithm settings 
%  options.noise        : initial value for output data noise variance
%  options.numIterations: maximum number of EM iterations desired
%  ss                   : sufficient statistics collected so far
%
% Outputs:
%
%  A data structure "result" containing the final converged
%  values of:    
%
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
%
%  And a 2nd data structure "ss" containing the  (updated) 
%  sufficient statistics:
%
%  ss.N            
%  ss.sum_x2      
%  ss.sum_y2    
%  ss.sum_yx    
%  ss.sum_xxT    
%  ss.sum_z2_mean 
%  ss.sum_zx   
%
%***************************************************************
function [result, ss] = vbls_online_p(x, y, options, ss)	

% Check that we have the inputs needed
%--------------------------------------
if nargin < 2
  error('Insufficient inputs to vbls function');
end

if (~exist('options')|isempty(options))
  % Default settings for EM algorithm
  options.noise         = 1;
  options.numIterations = 50;
end

d    = size(x,2); 
TINY = 1e-6;

% If first data point, then exit early
%------------------------------------------
if ss.N == 0 
   % Initialization of parameters  (No changes needed here) 
   ss.a_alpha_init = 1e-8;  				
   ss.b_alpha_init = 1e-8*ones(d,1);  	
   ss.a_alpha = ss.a_alpha_init;
   ss.b_alpha = ss.b_alpha_init;
   ss.alpha_mean = ss.a_alpha ./ ss.b_alpha;	
   ss.b_mean = zeros(d,1);		
   ss.b_var = ones(d,1) ./ ss.alpha_mean;  
   ss.psi_y = options.noise;	
   ss.psi_z = options.noise*ones(d,1);	 
  
   % Store parameter values in output result
   result.alpha_mean = ss.alpha_mean;
   result.a_alpha = ss.a_alpha;
   result.b_alpha = ss.b_alpha;
   result.b_mean = zeros(d,1); 
   result.b_var = ss.b_var;
   result.psi_y = ss.psi_y;
   result.psi_z = ss.psi_z;

   % Update sufficient statistics
   ss.N  = 1;
   ss.sum_x2 = (x.^2)';
   ss.sum_y2 = y^2;
   ss.sum_yx = y*x';
   ss.sum_xxT = x'*x;
   ss.sum_z2_mean = 0;
   ss.sum_zx = 0;

   return;
end

% Otherwise, this is not the 1st data sample ...
%--------------------------------------------------
old_b_mean = ss.b_mean;  % for recursive updates below

% Update sufficient statistics
ss.sum_x2  = ss.sum_x2 + (x.^2)';
ss.sum_y2  = ss.sum_y2 + (y^2);
ss.sum_yx  = ss.sum_yx + y*x';
ss.sum_xxT = ss.sum_xxT + x'*x;
ss.N       = ss.lambda * ss.N + 1;

% Use parameter values from previous time step
b_mean     = ss.b_mean;
b_var      = ss.b_var;
alpha_mean = ss.alpha_mean;
psi_y      = ss.psi_y;
psi_z      = ss.psi_z;

% EM Algorithm
%--------------
for numIter = 1:options.numIterations
   
   % E-step
   %--------
   % Compute mean (Nxd) and variance (dx1) of hidden variables, z
   s           = psi_y + sum(psi_z./alpha_mean);  
   z_var       = psi_z./alpha_mean - (psi_z./alpha_mean).^2/(s + TINY);
   z_mean      = ( x*diag(b_mean) + 1/(s + TINY)*(y - x*b_mean)*(psi_z./alpha_mean)' )';
   z2_mean     = z_mean.^2 + z_var;
   z_mean_1    = sum(psi_z./alpha_mean)/(s + TINY)*y ...
                  + (1 - 1/(s + TINY)*sum(psi_z./alpha_mean))*x*b_mean;
   one_z_var_one = sum(psi_z./alpha_mean)*(1 - 1/(s + TINY)*sum(psi_z./alpha_mean));

   % Update sufficient statistics
   sum_z2_mean = ss.lambda * ss.sum_z2_mean + (z_mean.^2);
   sum_zx      = ss.lambda * ss.sum_zx + (z_mean.*x');

   % Compute mean and variance of regression variable (both dx1 vectors)
   sum_x2_psiz = ss.sum_x2 + psi_z;  
   b_mean      = ss.sum_x2 ./ (sum_x2_psiz + TINY) .* old_b_mean ...
                 +  (psi_z ./ alpha_mean) .* ...
                    (ss.sum_yx - ss.sum_xxT*old_b_mean) ./ (s*sum_x2_psiz); 
   b_var       = (psi_z./alpha_mean)./(sum_x2_psiz + TINY);  
   b2_mean     = b_mean.^2 + b_var;
    
   % Calculate mean of precision variable (dx1 vector)
   a_alpha     = ss.a_alpha_init + ss.N/2;
   b_alpha     = ss.b_alpha_init + ...
                 0.5*(sum_z2_mean + ss.N*z_var - (sum_zx.^2)./sum_x2_psiz ) ...
                  ./(psi_z + TINY);
   alpha_mean  = a_alpha ./ b_alpha;
   
   % M-step
   %--------
   psi_y = (1 - sum(psi_z./alpha_mean)/(s + TINY))^2 * ...
           (ss.sum_y2 - 2*b_mean'*ss.sum_yx + ...
            b_mean'*ss.sum_xxT*b_mean) / ss.N + one_z_var_one;
   psi_z  = ((psi_z / s).^2) .* ...
           (ss.sum_y2 - 2*b_mean'*ss.sum_yx + ...
           b_mean'*ss.sum_xxT*b_mean) ./ alpha_mean ./ ss.N ...
          + alpha_mean .* z_var ...
          + alpha_mean .* b_var .* (ss.sum_x2 / ss.N);
end 

% Update the data structure with sufficient statistics
%-------------------------------------------------------
ss.sum_z2_mean      = sum_z2_mean;
ss.sum_zx           = sum_zx;

% Fill the output data structure 
%--------------------------------
result.alpha_mean       = alpha_mean;
result.a_alpha          = a_alpha;
result.b_alpha          = b_alpha;
result.b_mean           = b_mean;
result.b_var            = b_var;
result.psi_y            = psi_y;
result.psi_z            = psi_z;
result.z_var            = z_var;
result.z_mean           = z_mean;
result.numIter          = numIter;
