function [ QtT_sign_corrected ] = preprocessQuaternionSignal( QtT, omega_tT, dt )
    division_epsilon        = 1.0e-100;
    Qt_plus_1T              = integrateQuat( QtT, omega_tT, dt, 1.0 );
    QtT_prediction          = QtT;
    QtT_prediction(:,2:end) = Qt_plus_1T(:,1:end-1);
    sign_mismatch_per_Q_dim = ((QtT_prediction ./ (QtT + division_epsilon)) < 0);
    sign_mismatch_summary   = and(and(sign_mismatch_per_Q_dim(1,:), ...
                                      sign_mismatch_per_Q_dim(2,:)), ...
                                  and(sign_mismatch_per_Q_dim(3,:), ...
                                      sign_mismatch_per_Q_dim(4,:)));
    sign_mismatch_idx       = find(sign_mismatch_summary, 1);
    if (isempty(sign_mismatch_idx))
        QtT_sign_corrected  = QtT;
    else
        if (mod(length(sign_mismatch_idx), 2) == 1)
            sign_mismatch_idx   = [sign_mismatch_idx, size(QtT, 2)+1];
        end
        sign_corrector      = ones(size(QtT));
        for smii=1:2:length(sign_mismatch_idx)
            sign_corrector(:,sign_mismatch_idx(smii):(sign_mismatch_idx(smii+1)-1)) = -sign_corrector(:,sign_mismatch_idx(smii):(sign_mismatch_idx(smii+1)-1));
        end
        QtT_sign_corrected  = sign_corrector .* QtT;
    end
end