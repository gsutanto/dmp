function [ omegaT ] = computeOmegaTrajectory( QT, dt )
    % Author        : Giovanni Sutanto
    % Date          : March 2017
    % Description   : Given Quaternion trajectory and dt, 
    %                 compute the angular velocity (omega) trajectory.
    assert(size(QT, 1) == 4, 'Each column of QT must be a Quaternion (4-dimensional)!');
    
    QtT                     = QT;
    QtT(:,end)              = QtT(:,end-1);
    Qt_plus_1T              = QT;
    Qt_plus_1T(:,1:end-1)   = QT(:,2:end);
    omegaT                  = (1.0/dt) * computeLogQuatDifference(Qt_plus_1T, QtT);
end

