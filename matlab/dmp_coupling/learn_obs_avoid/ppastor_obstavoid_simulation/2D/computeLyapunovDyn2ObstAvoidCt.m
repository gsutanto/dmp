function [ Ct_obs ] = computeLyapunovDyn2ObstAvoidCt( betaDYN2, kDYN2, ox3, v3 )
    if ((norm(ox3) >= 1e-2) && (norm(v3) >= 1e-2))
        Ct_obs      = [-norm(v3)*((2*kDYN2*ox3)-((betaDYN2/(norm(v3)*norm(ox3)))*v3)+((betaDYN2/(norm(v3)*(norm(ox3))^3))*(v3.'*ox3)*ox3))*exp((((betaDYN2/(norm(v3)*norm(ox3)))*v3)-(kDYN2*(ox3))).'*ox3)];
    else
        Ct_obs      = zeros(3,1);
    end
end

