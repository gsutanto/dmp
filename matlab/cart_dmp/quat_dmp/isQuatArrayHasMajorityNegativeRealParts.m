function [ result ] = isQuatArrayHasMajorityNegativeRealParts( Qs )
    N_Quat  = size(Qs, 2);
    count_Quat_w_negative_real_parts    = 0;
    
    for i=1:N_Quat
        if (Qs(1,i) < 0.0)
            count_Quat_w_negative_real_parts    = count_Quat_w_negative_real_parts + 1;
        end
    end
    
    result  = (count_Quat_w_negative_real_parts > (N_Quat/2.0));
end

