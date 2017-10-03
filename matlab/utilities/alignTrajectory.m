function [ traj_s, traj_tau ] = alignTrajectory( ref_traj, traj, dt, varargin )
    % Author        : Giovanni Sutanto
    % Date          : April 2017
    % Description   : Align a trajectory w.r.t. reference trajectory, 
    %                 subject to time delay and time stretching,
    %                 using Dynamic Time Warping (DTW, dtw.m) 
    %                 for feature traj_correspondence matching.
    % Terminology   :
    %   s           : start time
    %   tau         : time stretch factor
    % Input         :  
    %   ref_traj    : reference trajectory (defined with ref_traj_s = 1 
    %                 and ref_traj_tau = 1)
    %   traj        : trajectory to be aligned
    %   dt          : time-step length (for filtering velocity signal 
    %                 of the trajectory, which is used for 
    %                 weighting least-squares on traj_correspondences)
    % Output        :
    %   traj_s      : the resulting/computed start time of the 
    %                 trajectory to be aligned
    %   traj_tau    : the resulting/computed time stretch factor of the 
    %                 trajectory to be aligned (relative to ref_traj_tau,
    %                 since ref_traj_tau == 1)
    if (nargin > 3)
        is_using_weighted_least_squares     = varargin{1};
    else
        is_using_weighted_least_squares     = 1;
    end
    
    if (nargin > 4)
        is_plotting_alignment               = varargin{2};
    else
        is_plotting_alignment               = 0;
    end
    
    if (nargin > 5)
        max_dtw_input_traj_length           = varargin{3};  % there is maximum length 
                                                            % (this is to bound the computation time of DTW, 
                                                            % but you will pay the price because 
                                                            % traj_s will NOT be as accurate as before due to rounding error!!!)
    else
        max_dtw_input_traj_length           = -1;           % no maximum length, retain original length
    end
    
    if (nargin > 6)
        N_match_display                     = varargin{4};
    else
        N_match_display                     = 8;
    end
    
    assert(dt > 0, 'dt must be > 0 !!!');
    assert(size(ref_traj,2) == size(traj,2), 'Dimensionality of the trajectories is NOT matched.');
    D       = size(ref_traj, 2);
    
    % the global weight of velocity correspondences (relative to position correspondences)
    velocity_correspondence_weight          = 1.0;
    
    corresp_velocity_squared_exp_distance_const     = 5.0;
    velocity_from_max_squared_exp_distance_const    = 2.5;
    
    corresp_acceleration_squared_exp_distance_const = 20.0;
    acceleration_from_max_squared_exp_distance_const= 10.0;
    
    fs      = 1.0/dt;               % sampling frequency = 1.0/dt
    % Position Low-Pass Filter Setup:
    fc      = 5.0;                  % cutoff frequency   = 5.0 Hz
    N_order = 2;
    [b, a] 	= butter(N_order, fc/(fs/2)); % low-pass N_order-th-order Butterworth filter
    
    % Velocity and Acceleration Low-Pass Filter Setup:
    fcd     = 1.0;                  % cutoff frequency   = 1.0 Hz
    N_orderd= 2;
    [bd, ad]= butter(N_orderd, fcd/(fs/2)); % low-pass N_orderd-th-order Butterworth filter
    
    percentage_zero_padding     = 3.0;
    
    % Velocity Trajectories:
    ref_trajd   = zeros(size(ref_traj));
    trajd       = zeros(size(traj));
    
    % Acceleration Trajectories:
    ref_trajdd  = zeros(size(ref_traj));
    trajdd      = zeros(size(traj));
    
    if (is_using_weighted_least_squares)
        % each traj_correspondence are to be weighted by
        % "variation" at the traj_correspondence coordinate of the trajectory
        % (because "flat" signal/trajectory may make erroneous traj_correspondences):
        traj_correspondences_weight             = [];
        traj_correspondences_dim_weight_cell    = cell(D,1);
        trajd_correspondences_dim_weight_cell   = cell(D,1);
    end
    
    if (max_dtw_input_traj_length > 0)
        if (min(size(ref_traj, 1), size(traj, 1)) > max_dtw_input_traj_length)
            is_using_global_traj_length_scale   = 1;
            global_traj_length_scale    = max_dtw_input_traj_length * 1.0 / (min(size(ref_traj, 1), size(traj, 1)));
            new_ref_traj_length         = round(global_traj_length_scale * size(ref_traj,1));
            new_traj_length             = round(global_traj_length_scale * size(traj,1));
            
            new_ref_traj    = zeros(new_ref_traj_length, D);
            new_traj        = zeros(new_traj_length, D);
            
            new_ref_trajd   = zeros(new_ref_traj_length, D);
            new_trajd       = zeros(new_traj_length, D);
            
            new_ref_trajdd  = zeros(new_ref_traj_length, D);
            new_trajdd      = zeros(new_traj_length, D);
        else
            is_using_global_traj_length_scale   = 0;
        end
    else
        is_using_global_traj_length_scale       = 0;
    end
    
    ref_traj_zero_padding_length        = round((percentage_zero_padding/100.0) * size(ref_traj, 1));
    traj_zero_padding_length            = round((percentage_zero_padding/100.0) * size(traj, 1));
    
    % compute traj_correspondences:
    traj_correspondences_dim_mat_cell   = cell(D,1);
    trajd_correspondences_dim_mat_cell  = cell(D,1);
    for d=1:D
        % compute and filter the (velocity) trajectories:
        ref_trajd(:,d)                  = diffnc(ref_traj(:,d), dt);
        trajd(:,d)                      = diffnc(traj(:,d), dt);
        
        % compute and filter the (acceleration) trajectories:
        ref_trajdd(:,d)                 = diffnc(ref_trajd(:,d), dt);
        trajdd(:,d)                     = diffnc(trajd(:,d), dt);
        
        % adding initial and end zero-padding to handle imperfect cut/
        % boundary conditions on velocity trajectories:
        ref_trajd(1:ref_traj_zero_padding_length,d)         = 0;
        ref_trajd(end-ref_traj_zero_padding_length:end,d)   = 0;
        trajd(1:traj_zero_padding_length,d)                 = 0;
        trajd(end-traj_zero_padding_length:end,d)           = 0;
        
        % adding initial and end zero-padding to handle imperfect cut/
        % boundary conditions on acceleration trajectories:
        ref_trajdd(1:ref_traj_zero_padding_length,d)        = 0;
        ref_trajdd(end-ref_traj_zero_padding_length:end,d)  = 0;
        trajdd(1:traj_zero_padding_length,d)                = 0;
        trajdd(end-traj_zero_padding_length:end,d)          = 0;
            
        % filter the (position) trajectories:
        ref_traj(:,d)                   = filtfilt(b, a, ref_traj(:,d));
        traj(:,d)                       = filtfilt(b, a, traj(:,d));
        
        % filter the (velocity) trajectories:
        ref_trajd(:,d)                  = filtfilt(bd, ad, ref_trajd(:,d));
        trajd(:,d)                      = filtfilt(bd, ad, trajd(:,d));
        
        % filter the (acceleration) trajectories:
        ref_trajdd(:,d)                 = filtfilt(bd, ad, ref_trajdd(:,d));
        trajdd(:,d)                     = filtfilt(bd, ad, trajdd(:,d));
        
        % make sure that the trajectories are zero-mean, such that the cost is
        % not dominated by the current cost:
        ref_traj(:,d)                   = (ref_traj(:,d)   - mean(ref_traj(:,d)))   /std(ref_traj(:,d));
        traj(:,d)                       = (traj(:,d)       - mean(traj(:,d)))       /std(traj(:,d));
        ref_trajd(:,d)                  = (ref_trajd(:,d)  - mean(ref_trajd(:,d)))  /std(ref_trajd(:,d));
        trajd(:,d)                      = (trajd(:,d)      - mean(trajd(:,d)))      /std(trajd(:,d));
        ref_trajdd(:,d)                 = (ref_trajdd(:,d) - mean(ref_trajdd(:,d))) /std(ref_trajdd(:,d));
        trajdd(:,d)                     = (trajdd(:,d)     - mean(trajdd(:,d)))     /std(trajdd(:,d));
        
        if (is_using_global_traj_length_scale)
            new_ref_traj(:,d)   = stretchTrajectory( ref_traj(:,d).', new_ref_traj_length ).';
            new_traj(:,d)       = stretchTrajectory( traj(:,d).', new_traj_length ).';
            
            new_ref_trajd(:,d)  = stretchTrajectory( ref_trajd(:,d).', new_ref_traj_length ).';
            new_trajd(:,d)      = stretchTrajectory( trajd(:,d).', new_traj_length ).';
            
            new_ref_trajdd(:,d) = stretchTrajectory( ref_trajdd(:,d).', new_ref_traj_length ).';
            new_trajdd(:,d)     = stretchTrajectory( trajdd(:,d).', new_traj_length ).';
        end
    end
    
    if (is_using_global_traj_length_scale)
        ref_traj    = new_ref_traj;
        traj        = new_traj;
        
        ref_trajd   = new_ref_trajd;
        trajd       = new_trajd;
        
        ref_trajdd  = new_ref_trajdd;
        trajdd      = new_trajdd;
    end
        
    for d=1:D
        % compute traj_correspondences of (position) trajectory:
        [~, ~, ~, traj_correspondences_dim]     = dtw(ref_traj(:,d), traj(:,d));
        temp_traj_correspondences_dim_mat       = cell2mat(traj_correspondences_dim);
        
        % compute traj_correspondences of (velocity) trajectory:
        [~, ~, ~, trajd_correspondences_dim]    = dtw(ref_trajd(:,d), trajd(:,d));
        temp_trajd_correspondences_dim_mat      = cell2mat(trajd_correspondences_dim);
        if (is_using_weighted_least_squares)
            %% Position Correspondences
            
            % minimum absolute (standardized) velocity associated with each
            % (position) trajectory traj_correspondences:
            min_abs_velocity_correspondences    = min(abs(ref_trajd(temp_traj_correspondences_dim_mat(:,1),d)), ...
                                                      abs(trajd(temp_traj_correspondences_dim_mat(:,2),d)));
                                                  
            % select only (position) trajectory traj_correspondences which have
            % (min_abs_velocity_correspondences > threshold_min_abs_velocity):
            max_min_abs_velocity                    = max(min_abs_velocity_correspondences);
            threshold_min_abs_velocity              = max_min_abs_velocity/2.5;
            traj_correspondences_dim_mat_pass_idx   = [find(min_abs_velocity_correspondences > threshold_min_abs_velocity)];
            traj_correspondences_dim_mat_cell{d,1}  = temp_traj_correspondences_dim_mat(traj_correspondences_dim_mat_pass_idx,:);
            
            % measure distance in velocity (between corresponded positions, in reference and target position trajectories):
            corresp_velocity_squared_exp_distance   = exp(-corresp_velocity_squared_exp_distance_const * ...
                                                            ((ref_trajd(traj_correspondences_dim_mat_cell{d,1}(:,1),d) - ...
                                                              trajd(traj_correspondences_dim_mat_cell{d,1}(:,2),d)) ./ ...
                                                             min_abs_velocity_correspondences(traj_correspondences_dim_mat_pass_idx)) .^ 2);
            
            % measure distance between velocity and max velocity across
            % all position correpondences:
            range_velocity                          = max_min_abs_velocity - threshold_min_abs_velocity;
            velocity_from_max_squared_exp_distance  = exp(-velocity_from_max_squared_exp_distance_const * ...
                                                            ((max_min_abs_velocity - ...
                                                              min_abs_velocity_correspondences(traj_correspondences_dim_mat_pass_idx)) / ...
                                                             range_velocity) .^ 2);
            
            traj_correspondences_dim_weight_cell{d,1}   = velocity_from_max_squared_exp_distance .* ...
                                                          corresp_velocity_squared_exp_distance;
            
            %% Velocity Correspondences
            
            % minimum absolute (standardized) acceleration associated with each
            % (velocity) trajectory traj_correspondences:
            min_abs_acceleration_correspondences    = min(abs(ref_trajdd(temp_trajd_correspondences_dim_mat(:,1),d)), ...
                                                          abs(trajdd(temp_trajd_correspondences_dim_mat(:,2),d)));
            
            % select only (velocity) trajectory traj_correspondences which have
            % (min_abs_acceleration_correspondences > threshold_min_abs_acceleration):
            max_min_abs_acceleration                = max(min_abs_acceleration_correspondences);
            threshold_min_abs_acceleration          = max_min_abs_acceleration/2.5;
            trajd_correspondences_dim_mat_pass_idx  = [find(min_abs_acceleration_correspondences > threshold_min_abs_acceleration)];
            trajd_correspondences_dim_mat_cell{d,1} = temp_trajd_correspondences_dim_mat(trajd_correspondences_dim_mat_pass_idx,:);
            
            % measure distance in acceleration (between corresponded velocities, in reference and target velocity trajectories):
            corresp_acceleration_squared_exp_distance   = exp(-corresp_acceleration_squared_exp_distance_const * ...
                                                                ((ref_trajdd(trajd_correspondences_dim_mat_cell{d,1}(:,1),d) - ...
                                                                  trajdd(trajd_correspondences_dim_mat_cell{d,1}(:,2),d)) ./ ...
                                                                 min_abs_acceleration_correspondences(trajd_correspondences_dim_mat_pass_idx)) .^ 2);
            
            % measure distance between acceleration and max acceleration across
            % all velocity correpondences:
            range_acceleration                  = max_min_abs_acceleration - threshold_min_abs_acceleration;
            acceleration_from_max_squared_exp_distance  = exp(-acceleration_from_max_squared_exp_distance_const * ...
                                                                ((max_min_abs_acceleration - ...
                                                                  min_abs_acceleration_correspondences(trajd_correspondences_dim_mat_pass_idx)) / ...
                                                                 range_acceleration) .^ 2);
            
            trajd_correspondences_dim_weight_cell{d,1}  = acceleration_from_max_squared_exp_distance .* ...
                                                          corresp_acceleration_squared_exp_distance;
        else
            traj_correspondences_dim_mat_cell{d,1}  = temp_traj_correspondences_dim_mat;
            trajd_correspondences_dim_mat_cell{d,1} = temp_trajd_correspondences_dim_mat;
        end
    end
    traj_correspondences            = cell2mat(traj_correspondences_dim_mat_cell);
    trajd_correspondences           = cell2mat(trajd_correspondences_dim_mat_cell);
    if (is_using_weighted_least_squares)
        traj_correspondences_weight = cell2mat(traj_correspondences_dim_weight_cell);
        trajd_correspondences_weight= cell2mat(trajd_correspondences_dim_weight_cell);
    end
    N_traj_correspondences  = size(traj_correspondences, 1);
    N_trajd_correspondences = size(trajd_correspondences, 1);
    
    ref_traj_s              = 1;
    ref_traj_tau            = 1;
    
    % setup matrix A and b for weighted least-squares problem WAx == Wb
    A_traj  = ones(N_traj_correspondences, 2);
    b_traj  = ones(N_traj_correspondences, 1);
    for i=1:N_traj_correspondences
        traj_correspondence = traj_correspondences(i,:);
        ref_traj_t          = traj_correspondence(1,1);
        traj_t              = traj_correspondence(1,2);
        t                   = (ref_traj_t - ref_traj_s)/ref_traj_tau;
        A_traj(i,1)         = t;
        b_traj(i,1)         = traj_t;
    end
    A_trajd = ones(N_trajd_correspondences, 2);
    b_trajd = ones(N_trajd_correspondences, 1);
    for i=1:N_trajd_correspondences
        trajd_correspondence= trajd_correspondences(i,:);
        ref_traj_t          = trajd_correspondence(1,1);
        traj_t              = trajd_correspondence(1,2);
        t                   = (ref_traj_t - ref_traj_s)/ref_traj_tau;
        A_trajd(i,1)        = t;
        b_trajd(i,1)        = traj_t;
    end
    A   = [A_traj; A_trajd];
    b   = [b_traj; b_trajd];
    
    if (is_using_weighted_least_squares)
%         % normalize the position correspondences weights:
%         sum_W_traj      = sum(traj_correspondences_weight);
%         W_traj_vector   = (1.0/sum_W_traj) * traj_correspondences_weight;
%         
%         % normalize the velocity correspondences weights:
%         sum_W_trajd     = sum(trajd_correspondences_weight);
%         W_trajd_vector  = (1.0/sum_W_trajd) * trajd_correspondences_weight;
        
        % concatenate position correspondences with 
        % weighted(-down) velocity correspondences:
%         W_vector        = [W_traj_vector; velocity_correspondence_weight * W_trajd_vector];
        W_vector        = [traj_correspondences_weight; velocity_correspondence_weight * trajd_correspondences_weight];
        W               = (1.0/(sum(W_vector))) * diag(W_vector);
    end
    
    if (is_using_weighted_least_squares)
        % perform weighted least-squares WAx == Wb, with x == [traj_tau; traj_s]
        x       = inv(A.' * W * A) * A.' * W * b;
    else
        x       = A \ b;
    end
    traj_tau    = x(1,1);
    traj_s      = x(2,1);
    
    % measure fitting performance/quality by WNMSE (if weighted) or 
    % NMSE (if unweighted):
    if (is_using_weighted_least_squares)
        % compute Weighted Mean Squared Error:
        WMSE        = mean( W * (((A*x)-b).^2) );

        % compute Weighted Variance:
        mean_b      = mean(b);
        zero_mean_b = b - mean_b;
        WVar        = (1.0/(N_traj_correspondences-1)) * zero_mean_b.' * W * zero_mean_b;

        % compute Weighted Normalized Mean Squared Error (WNMSE):
        WNMSE       = WMSE/WVar;
        fprintf('WNMSE Alignment = %f\n', WNMSE);

%         if (WNMSE > 0.15)
%             keyboard;
%         end
    else
        % compute Normalized Mean Squared Error (NMSE):
        [~, NMSE, ~]= computeNMSE( A, x, b );
        fprintf('NMSE Alignment = %f\n', NMSE);

%         if (NMSE > 0.15)
%             keyboard;
%         end
    end
    
    if (is_using_global_traj_length_scale)
        % rescale back traj_s:
        traj_s  = traj_s / global_traj_length_scale;
    end
    
    if (is_plotting_alignment)
        match_display_traj_indices      = [1:1+floor((size(traj_correspondences_dim_mat_cell{d,1},1)-(N_match_display-2)-1)/(N_match_display-1)):size(traj_correspondences_dim_mat_cell{d,1},1)].';
        if (length(match_display_traj_indices) ~= N_match_display)
            match_display_traj_indices  = [match_display_traj_indices; size(traj_correspondences_dim_mat_cell{d,1},1)];
        end
        
        match_display_trajd_indices = [1:1+floor((size(trajd_correspondences_dim_mat_cell{d,1},1)-(N_match_display-2)-1)/(N_match_display-1)):size(trajd_correspondences_dim_mat_cell{d,1},1)].';
        if (length(match_display_trajd_indices) == (N_match_display - 1))
            match_display_trajd_indices = [match_display_trajd_indices; size(trajd_correspondences_dim_mat_cell{d,1},1)];
        elseif (length(match_display_trajd_indices) == (N_match_display + 1))
            match_display_trajd_indices = match_display_trajd_indices(1:N_match_display);
        elseif (length(match_display_trajd_indices) ~= N_match_display)
            warning('Length of match_display_trajd_indices is incorrect!!!');
            keyboard;
            error('Length of match_display_trajd_indices is incorrect!!!');
        end
        
        if (is_using_weighted_least_squares)
            [~, sorted_traj_idx]        = sort(traj_correspondences_dim_weight_cell{d,1},'descend');
            top_weighted_traj_indices   = sorted_traj_idx(match_display_traj_indices,1);
            match_display_traj_indices  = top_weighted_traj_indices;
            
            [~, sorted_trajd_idx]       = sort(trajd_correspondences_dim_weight_cell{d,1},'descend');
            top_weighted_trajd_indices  = sorted_trajd_idx(match_display_trajd_indices,1);
            match_display_trajd_indices = top_weighted_trajd_indices;
        end
        
        % data indices
        data_ID     = cell(0);
        for nmd = 1:1:N_match_display
            data_ID{1, nmd} = num2str(nmd);
        end
        
        for d=1:D
            figure;
            subplot(2, 1, 1);
                hold on;
                    plot(ref_traj(:,d), 'r');
                    legend('reference');
                    title(['Alignment of Position Trajectory dimension ', num2str(d)]);
                    match_disp_x    = traj_correspondences_dim_mat_cell{d,1}(match_display_traj_indices,1);
                    plot(match_disp_x,ref_traj(match_disp_x,d),'g.');
                    text(match_disp_x,ref_traj(match_disp_x,d)+0.005,data_ID);
                hold off;
            subplot(2, 1, 2);
                hold on;
                    plot(traj(:,d), 'b');
                    legend('target');
                    match_disp_x    = traj_correspondences_dim_mat_cell{d,1}(match_display_traj_indices,2);
                    plot(match_disp_x,traj(match_disp_x,d),'g.');
                    text(match_disp_x,traj(match_disp_x,d)+0.005,data_ID);
                hold off;
            
            figure;
            subplot(2, 1, 1);
                hold on;
                    plot(ref_trajd(:,d), 'r');
                    legend('reference');
                    title(['Alignment of Velocity Trajectory dimension ', num2str(d)]);
                    match_disp_x    = trajd_correspondences_dim_mat_cell{d,1}(match_display_trajd_indices,1);
                    plot(match_disp_x,ref_trajd(match_disp_x,d),'g.');
                    text(match_disp_x,ref_trajd(match_disp_x,d)+0.005,data_ID);
                hold off;
            subplot(2, 1, 2);
                hold on;
                    plot(trajd(:,d), 'b');
                    legend('target');
                    match_disp_x    = trajd_correspondences_dim_mat_cell{d,1}(match_display_trajd_indices,2);
                    plot(match_disp_x,trajd(match_disp_x,d),'g.');
                    text(match_disp_x,trajd(match_disp_x,d)+0.005,data_ID);
                hold off;
        end
        keyboard;
    end
end