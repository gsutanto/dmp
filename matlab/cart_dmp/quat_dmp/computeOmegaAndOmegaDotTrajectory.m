function [ omegaT, omegadT ] = computeOmegaAndOmegaDotTrajectory( QT, QTd, QTdd )
    % Author        : Giovanni Sutanto
    % Date          : December 2016
    % Description   : Extracting/converting omega and omegad (trajectories) 
    %                 from trajectories of Q, Qd, and Qdd.
    QT_conj     = computeQuatConjugate(QT);
    omegaQT     = (2.0) * computeQuatProduct( QTd, QT_conj );
    omegadQT    = (2.0) * computeQuatProduct( ...
                            (QTdd - computeQuatProduct( ...
                                        QTd, computeQuatProduct( ...
                                                QT_conj, QTd ) )), QT_conj );
    
    % some anomaly-checking:
    if (norm(omegaQT(1,:)) > 0)
        fprintf('WARNING: norm(omegaQT(1,:))        = %f > 0\n', norm(omegaQT(1,:)));
        fprintf('         max(abs(omegaQT(1,:)))    = %f\n', max(abs(omegaQT(1,:))));
    end
    if (norm(omegadQT(1,:)) > 0)
        fprintf('WARNING: norm(omegadQT(1,:))       = %f > 0\n', norm(omegadQT(1,:)));
        fprintf('         max(abs(omegadQT(1,:)))   = %f\n', max(abs(omegadQT(1,:))));
    end
    
    omegaT  = omegaQT(2:4,:);
    omegadT = omegadQT(2:4,:);
end

