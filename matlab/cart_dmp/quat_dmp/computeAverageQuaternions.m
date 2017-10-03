function [ mean_Q ] = computeAverageQuaternions( Qs, varargin )
    % Author        : Giovanni Sutanto
    % Date          : December 2016
    
    if (nargin == 2)
        method  = varargin{1};
    else
        method  = 1;    % default
    end
    
    Qs  = normalizeQuaternion(Qs);
    
    if (method == 1)
        QsQsT   = Qs * Qs';
        [V, D]  = eig(QsQsT);
        eigvals = diag(D);
        [~, max_eig_val_idx]    = max(eigvals);
        max_eig_vec             = V(:, max_eig_val_idx);
        mean_Q                  = normalizeQuaternion(max_eig_vec);
    elseif (method == 2)
        % this method is still experimental
        Q_anchor                = Qs(:,1);
        N_samples               = size(Qs,2);
        sum_log_Q_diff          = zeros(3,1);
        for i=2:N_samples
            log_Q_diff      = computeLogMapQuat(computeQuatProduct(computeQuatConjugate(Q_anchor), Qs(:,i)));
            sum_log_Q_diff  = sum_log_Q_diff + log_Q_diff;
        end
        scaled_sum_log_Q_diff   = (1/N_samples) * sum_log_Q_diff;
        mean_Q                  = normalizeQuaternion(...
                                    computeQuatProduct(Q_anchor, ...
                                                       computeExpMapQuat(...
                                                            scaled_sum_log_Q_diff)));
    end
end