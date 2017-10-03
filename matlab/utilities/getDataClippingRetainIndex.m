function [ retain_idx_cell ] = getDataClippingRetainIndex( ...
                                    traj_input, varargin )
% Author        : Giovanni Sutanto
% Date          : Feb. 1, 2017
% Description   : Given end-effector velocity trajectory (dimension N X D),
%                 return data indices cell, with each cell component being
%                 a range of index denoting a trajectory from which
%                 a movement primitive could be learned.
    if (nargin > 1)
        is_plotting = varargin{1};
    else
        is_plotting = 0;    % default is not plotting the clipping points
    end
    if (nargin > 2)
        motion_start_and_end_zero_crossing_trimming_threshold   = varargin{2};
    else
        motion_start_and_end_zero_crossing_trimming_threshold   = 0.12;
    end
    if (nargin > 3)
        motion_segment_zero_crossing_trimming_threshold     = varargin{3};
    else
        motion_segment_zero_crossing_trimming_threshold     = 0.035;
    end
    if (nargin > 4)
        N_motion_segments   = varargin{4};
    else
        N_motion_segments   = 3;
    end
    if (nargin > 5)
        dt                  = varargin{5};
    else
        dt                  = 1.0/300.0;    % sampling time of SL xrarm (ARM robot)
    end
    if (nargin > 6)
        is_traj_input_velocity  = varargin{6};
    else
        is_traj_input_velocity  = 1;
    end
    if (nargin > 7)
        is_only_computing_1st_zero_crossing     = varargin{7};
    else
        is_only_computing_1st_zero_crossing     = 0;
    end
    if (nargin > 8)
        fc              = varargin{8};
    else
        fc              = 5.0;
    end
    if (nargin > 9)
        is_first_trimming_point_maxima  = varargin{9};  % First trimming point is a maxima? This is especially the case for initial impact of tactile sensing.
    else
        is_first_trimming_point_maxima  = 0;
    end
    
    fs                  = 1/dt;
    N_filter_order      = 2;
    [b, a]              = butter(N_filter_order, fc/(fs/2));
    
    if (is_traj_input_velocity)
        traj_velocity   = traj_input;
    else
        traj_velocity   = zeros(size(traj_input));
        for d=1:size(traj_input, 2)
            temp                = diffnc(traj_input(:,d), dt);
            temp(1:20,1)        = 0;
            traj_velocity(:,d)  = filtfilt(b, a, temp);
        end
    end
    
    norm_speed          = sqrt(sum(traj_velocity.^2, 2));
    
    filted_norm_speed   = filtfilt(b, a, norm_speed);
    
    [potential_trim_idx]= crossing(filted_norm_speed,[],...
                                   motion_start_and_end_zero_crossing_trimming_threshold);
    trim_idx            = [min(potential_trim_idx), max(potential_trim_idx)];
    
    % refine index on the filtered norm-speed trajectory
    if (is_first_trimming_point_maxima)
        [ trim_idx(1,1), ~ ]= getLocalMaxima( filted_norm_speed, trim_idx(1,1) );
    else
        [ trim_idx(1,1), ~ ]= getLocalMinima( filted_norm_speed, trim_idx(1,1) );
    end
    if (is_only_computing_1st_zero_crossing == 0)
        [ trim_idx(1,2), ~ ]= getLocalMinima( filted_norm_speed, trim_idx(1,2) );
    end
    
    % refine index on the original norm-speed trajectory
    if (is_first_trimming_point_maxima)
        [ trim_idx(1,1), ~ ]= getLocalMaxima( norm_speed, trim_idx(1,1) );
    else
        [ trim_idx(1,1), ~ ]= getLocalMinima( norm_speed, trim_idx(1,1) );
    end
    if (is_only_computing_1st_zero_crossing == 0)
        [ trim_idx(1,2), ~ ]= getLocalMinima( norm_speed, trim_idx(1,2) );
    end
    
    if (N_motion_segments > 1)
        trimmed_filted_norm_speed    = filted_norm_speed((trim_idx(1,1)+1):trim_idx(1,2), :);
        [potential_segment_idx] = crossing(trimmed_filted_norm_speed,[],...
                                           motion_segment_zero_crossing_trimming_threshold);
        segment_idx     = (trim_idx(1,1)) + [min(potential_segment_idx), max(potential_segment_idx)];
        
        % refine index on the filtered norm-speed trajectory
        [ segment_idx(1,1), ~ ] = getLocalMinima( filted_norm_speed, segment_idx(1,1) );
        [ segment_idx(1,2), ~ ] = getLocalMinima( filted_norm_speed, segment_idx(1,2) );
        
        % refine index on the original norm-speed trajectory
        [ segment_idx(1,1), ~ ] = getLocalMinima( norm_speed, segment_idx(1,1) );
        [ segment_idx(1,2), ~ ] = getLocalMinima( norm_speed, segment_idx(1,2) );
    end
    
    if (is_plotting)
        figure;
        hold on;
            pnes    = plot(norm_speed,'b');
            pfnes   = plot(filted_norm_speed,'r');
            pt      = plot(trim_idx, 0, 'ko',...
                           'LineWidth',3, 'MarkerSize', 10,...
                           'MarkerFaceColor','k');
            if (N_motion_segments > 1)
                ps  = plot(segment_idx, 0, 'ro',...
                           'LineWidth',3, 'MarkerSize', 10,...
                           'MarkerFaceColor','r');
                legend([pnes, pfnes, pt(1), ps(1)], ...
                       'norm\_endeff\_speed',...
                       'filtered\_norm\_endeff\_speed',...
                       'trim\_index',...
                       'segment\_index');
            else
                legend([pnes, pfnes, pt(1)], ...
                       'norm\_endeff\_speed',...
                       'filtered\_norm\_endeff\_speed',...
                       'trim\_index');
            end
        hold off;
        keyboard;
    end
    
    if (N_motion_segments > 1)
        retain_idx_cell     = { [trim_idx(1,1):segment_idx(1,1)],...
                                [(segment_idx(1,1)+1):(segment_idx(1,2)-1)],...
                                [segment_idx(1,2):trim_idx(1,2)]};
    else
        retain_idx_cell     = { [trim_idx(1,1):trim_idx(1,2)] };
    end
end

