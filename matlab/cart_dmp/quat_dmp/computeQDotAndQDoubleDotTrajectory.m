function [ QdT, QddT ] = computeQDotAndQDoubleDotTrajectory( QT, omegaT, omegadT )
    % Author        : Giovanni Sutanto
    % Date          : December 2016
    % Description   : Extracting/converting Qd and Qdd (trajectories) 
    %                 from trajectories of Q, omega, and omegad.
    QdT     = 0.5 * computeQuatProduct([zeros(1,size(QT,2)); omegaT], QT);
    QddT    = 0.5 * (computeQuatProduct([zeros(1,size(QT,2)); omegadT], QT) + ...
                     computeQuatProduct([zeros(1,size(QdT,2)); omegaT], QdT));
end

