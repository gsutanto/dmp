function [ Ct_obs ] = computeParkHumanoids2008ObstAvoidCt( lambdaDYN1, betaDYN1, ox3, v3 )
    if ((norm(v3)>=0.0000001) || (norm(ox3)>=0.001))
        cos_theta   = v3.'*ox3/(norm(v3)*norm(ox3));
        if (cos_theta>0.0)
            Ct_obs  = lambdaDYN1*(cos_theta^(betaDYN1-1))*(1.0/(norm(ox3)^4))*(((v3.'*(ox3))*(-ox3))+(betaDYN1*[(-ox3),v3]*[v3.'*ox3; (norm(ox3)^2)]));
        else
            Ct_obs  = zeros(3,1);
        end
    else
        Ct_obs      = zeros(3,1);
    end
end

