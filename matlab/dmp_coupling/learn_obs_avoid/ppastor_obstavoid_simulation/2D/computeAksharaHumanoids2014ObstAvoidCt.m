function [ Ct_obs ] = computeAksharaHumanoids2014ObstAvoidCt( gamma, beta, k, ox3, v3 )
    % also might be sometimes referred as Humanoids'14 ObstAvoid Ct
    
    vl = norm(v3);
    if vl>0
        v3n = v3/vl;
    else
        v3n = v3;
    end

    x3 = -ox3;
    x3 = x3/norm(x3);
    rotaxis = cross(v3n,-x3);
    if vl>0
        rotaxis = rotaxis/norm(rotaxis);
    end
    R = rotmatrix(rotaxis,-pi/2);

    phi = acos(-v3n'*x3);

    dphi = gamma*phi*exp(-beta*phi)*exp(-k*(ox3.')*ox3);

    Ct_obs = dphi * R * v3;
end

