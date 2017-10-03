function [ smoothed_1D_position_prof ] = smoothStartEnd1DPositionProfile( oneD_position_prof, ...
                                                                          percentage_padding, ...
                                                                          percentage_smoothing_points, ...
                                                                          mode, ...
                                                                          b, a )    % filter coefficients
    traj_length             = length(oneD_position_prof);
    
    num_padding             = round((percentage_padding/100.0) * traj_length);
    if (num_padding <= 2)
        num_padding         = 3;                % minimum number of padding
    end
    
    num_smoothing_points    = round((percentage_smoothing_points/100.0) * traj_length);
    if (num_smoothing_points <= (num_padding+2))
        num_smoothing_points= (num_padding+3);  % minimum number of smoothing points
    end
    
    smoothed_1D_position_prof           = oneD_position_prof;
    if ((mode >= 1) && (mode <= 3))
        assert(num_padding > 2, 'num_padding must be greater than 2!');
        assert(num_smoothing_points > (num_padding+2), '# of smoothing points must be greater than (num_padding+2)!');
        assert(length(size(smoothed_1D_position_prof)) == 2, 'Input tensor must be 2-dimensional');
        assert(min(size(smoothed_1D_position_prof)) == 1, 'Input matrix must be 1-dimensional, i.e. a vector!');
    end
    
    % mode == 1: smooth start only
    % mode == 2: smooth end only
    % mode == 3: smooth both start and end
    % otherwise: no smoothing
    
    end_idx     = length(smoothed_1D_position_prof);
    
    if ((mode == 1) || (mode == 3))
        smoothed_1D_position_prof(2:num_padding)= smoothed_1D_position_prof(1);
        smoothed_1D_position_prof_idx           = [1:num_padding, (num_smoothing_points+1):end_idx];
        interp_position_prof_idx                = [(num_padding+1):num_smoothing_points];

        smoothed_1D_position_prof(interp_position_prof_idx) = interp1(smoothed_1D_position_prof_idx,...
                                                                      smoothed_1D_position_prof(smoothed_1D_position_prof_idx),...
                                                                      interp_position_prof_idx,...
                                                                      'linear');
    end
    
    if ((mode == 2) || (mode == 3))
        smoothed_1D_position_prof((end_idx-num_padding+1):(end_idx-1))  = smoothed_1D_position_prof(end_idx);
        smoothed_1D_position_prof_idx   = [1:(end_idx-num_smoothing_points), (end_idx-num_padding+1):end_idx];
        interp_position_prof_idx        = [(end_idx-num_smoothing_points+1):(end_idx-num_padding)];
        
        smoothed_1D_position_prof(interp_position_prof_idx) = interp1(smoothed_1D_position_prof_idx,...
                                                                      smoothed_1D_position_prof(smoothed_1D_position_prof_idx),...
                                                                      interp_position_prof_idx,...
                                                                      'linear');
    end
    
    % apply low-pass filter for smoothness:
    smoothed_1D_position_prof           = filtfilt(b, a, smoothed_1D_position_prof);
end