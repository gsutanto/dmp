function [ Qt ] = inverseIntegrateQuat( Qt_plus_1, omega_t, dt )
    % Author    : Giovanni Sutanto
    % Date      : February 2017
    
    assert(size(Qt_plus_1,2) == 1,'Currently inverseIntegrateQuat() function only supports 1-dimensional problem.');

    theta_v     = omega_t * (dt);
    Q_incr      = computeExpMapQuat(theta_v);
    Q_decr      = computeQuatConjugate( Q_incr );
    Qt          = computeQuatProduct(Q_decr, Qt_plus_1);
    Qt          = normalizeQuaternion(Qt);
end
