function [ Ct_obs ] = computeLyapunovGaussianObstAvoidCt( sigsq, ox3 )
    Ct_obs  = [-2*sigsq*exp(-sigsq*ox3.'*ox3)*ox3];
end

