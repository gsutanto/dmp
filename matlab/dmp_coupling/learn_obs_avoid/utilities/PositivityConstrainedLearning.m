function [ varargout ] = PositivityConstrainedLearning( varargin )
    phi = varargin{1};
    ct  = varargin{2};
    
    phiTphi     = phi.' * phi;
    phiTct      = phi.' * ct;

    w = quadprog(phiTphi, (-2*phiTct),[],[],[],[],...
                 zeros(size(phiTct, 1), 1), Inf(size(phiTct,1), 1));

    varargout(1) = {w};
end