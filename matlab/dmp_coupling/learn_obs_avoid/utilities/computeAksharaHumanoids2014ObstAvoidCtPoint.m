function [ Ct_obs ] = computeAksharaHumanoids2014ObstAvoidCtPoint( beta, k, ox3, v3 )
    % also might be sometimes referred as Humanoids'14 ObstAvoid Ct
    
    num_thresh  = 1e-2; % numerical threshold
    
    if ((norm(v3) > num_thresh) && (norm(ox3) > num_thresh))
        v3n     = v3/norm(v3);
        ox3n    = ox3/norm(ox3);
        rotaxis = cross(ox3n,v3n);
        if (norm(rotaxis) > num_thresh)
            rotaxis     = rotaxis/norm(rotaxis);
            R           = rotmatrix(rotaxis,pi/2);

            phi         = acos(ox3n.'*v3n);

            dphi        = phi * exp(-beta*phi) * exp(-k*(ox3.'*ox3));

            Ct_obs      = R * v3 * dphi;
        else
            Ct_obs      = zeros(3,1);
        end
    else
        Ct_obs = zeros(3,1);
    end
end

