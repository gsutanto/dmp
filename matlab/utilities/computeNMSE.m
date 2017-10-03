function [ varargout ] = computeNMSE( varargin )
    % for problem [X*w == b]
    if (nargin == 3)
        X           = varargin{1};
        w_fit       = varargin{2};
        b_target    = varargin{3};
        
        b_result    = X * w_fit;
    elseif (nargin == 2)
        b_result    = varargin{1};
        b_target    = varargin{2};
    end
    var_b_target    = var(b_target);
    
    mse             = mean( (b_result-b_target).^2 );
    nmse            = mse./var_b_target;
    
    varargout(1)    = {mse};
    varargout(2)    = {nmse};
    varargout(3)    = {b_result};
end