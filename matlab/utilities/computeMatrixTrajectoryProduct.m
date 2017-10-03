function [ C ] = computeMatrixTrajectoryProduct( A, B, varargin )
    % Author        : Giovanni Sutanto
    % Email         : gsutanto@usc.edu
    % Date          : 16 December 2016
    % Description   : C = A * B
    %                 Both A(:,i,:) and B(:,j,:) are matrices.
    assert(size(A,2) == size(B,2),'Trajectory length is NOT equal.');
    
    if (nargin == 3)
        method  = varargin{1};
    else
        method  = 1;        % default is using 'repmat' method
    end
    
    C           = zeros(size(A,1), size(A,2), size(B,3));
    A_transpose = permute(A, [3,2,1]);
    if (method == 1)        % using repmat
        for i=1:size(A,1)
            C(i,:,:)= sum((repmat(A_transpose(:,:,i),1,1,size(B,3)) .* B),1);
        end
    elseif (method == 2)    % using iterations
        for i=1:size(A,1)
            for j=1:size(B,3)
                C(i,:,j)= sum((A_transpose(:,:,i) .* B(:,:,j)),1);
            end
        end
    end
end

