function [ Qt_plus_1 ] = integrateQuat( Qt, omega, dt, tau )
    % Author    : Giovanni Sutanto
    % Date      : November 24, 2016
    assert(norm(Qt) > 0,'ERROR: integrateQuat(): norm(Qt) == 0!');

    theta_v     = (1/2) * omega * (dt/tau);
    Q_incr      = computeExpMapQuat(theta_v);
    Qt_plus_1   = normalizeQuaternion(computeQuatProduct(Q_incr, normalizeQuaternion(Qt)));
end
