function [ smoothed_Quat_traj ] = smoothStartEndQuatTrajectoryBasedOnQuaternion( Quat_traj, ...
                                                                                 percentage_padding, ...
                                                                                 percentage_smoothing_points, ...
                                                                                 mode, dt, ...
                                                                                 varargin )
    if (nargin > 5)
        fc  = varargin{1};
    else
        fc  = 40;           % 40 Hz cutoff frequency
    end
    if (nargin > 6)
        is_plotting_smoothing_comparison    = varargin{2};
    else
        is_plotting_smoothing_comparison    = 0;
    end
    
    fs      = 1/dt;                 % sampling frequency
    [b, a]  = butter(2, fc/(fs/2)); % 2nd order Butterworth filter
    
    Quat_prof               = Quat_traj{1,1};
    log_Quat_prof           = computeLogMapQuat(Quat_prof);
    smoothed_log_Quat_prof  = smoothStartEndNDPositionProfile( log_Quat_prof.', ...
                                                               percentage_padding, ...
                                                               percentage_smoothing_points, ...
                                                               mode, b, a ).';
    smoothed_Quat_prof      = computeExpMapQuat(smoothed_log_Quat_prof);
    
    traj_length             = size(smoothed_Quat_prof, 2);
    
    omega_prof              = computeOmegaTrajectory( smoothed_Quat_prof, dt );
    
    smoothed_omega_prof     = zeros(3, traj_length);
    smoothed_omegad_prof    = zeros(3, traj_length);
    
    for d=1:3
        smoothed_omega_prof(d,:)    = filtfilt(b, a, omega_prof(d,:).').';
        smoothed_omegad_prof(d,:)   = diffnc(smoothed_omega_prof(d,:).', dt).';
    end
    
    smoothed_Quat_traj      = cell(3,1);
    smoothed_Quat_traj{1,1} = smoothed_Quat_prof;
    smoothed_Quat_traj{2,1} = smoothed_omega_prof;
    smoothed_Quat_traj{3,1} = smoothed_omegad_prof;
    
    if (is_plotting_smoothing_comparison)
        figure;
        for d=1:4
            subplot(4,1,d);
            hold on
                plot(Quat_traj{1,1}(d,:),'r');
                plot(smoothed_Quat_traj{1,1}(d,:),'b');
                title('Quaternion');
                legend('original', 'smoothed');
            hold off
        end

        figure;
        for d=1:3
            subplot(3,1,d);
            hold on
                plot(log_Quat_prof(d,:),'r');
                plot(smoothed_log_Quat_prof(d,:),'b');
                title('log(Quat)');
                legend('original', 'smoothed');
            hold off
        end

        figure;
        for d=1:3
            subplot(3,1,d);
            hold on
                plot(Quat_traj{2,1}(d,:),'r');
                plot(smoothed_Quat_traj{2,1}(d,:),'b');
                title('Angular Velocity (Omega)');
                legend('original', 'smoothed');
            hold off
        end

        figure;
        for d=1:3
            subplot(3,1,d);
            hold on
                plot(Quat_traj{3,1}(d,:),'r');
                plot(smoothed_Quat_traj{3,1}(d,:),'b');
                title('Angular Acceleration (Omegadot)');
                legend('original', 'smoothed');
            hold off
        end
        
        keyboard;
    end
end