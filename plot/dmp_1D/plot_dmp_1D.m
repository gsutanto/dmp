function [  ] = plot_dmp_1D( varargin )
    % a simple MATLAB script to plot recorded DMP 1D trajectory
    % Author: Giovanni Sutanto
    % Date  : June 16, 2015
    close all;
    
    is_plot_goal_trajectory                         = 0;
    is_plot_canonical_sys                           = 0;
    is_plot_forcing_term                            = 0;
    is_plot_forcing_term_vs_canonical_state         = 0;
    if (nargin > 0)
        is_plot_goal_trajectory                     = varargin{1};
    end
    if (nargin > 1)
        is_plot_canonical_sys                       = varargin{2};
    end
    if (nargin > 2)
        is_plot_forcing_term                        = varargin{3};
    end
    if (nargin > 3)
        is_plot_forcing_term_vs_canonical_state     = varargin{4};
    end
    
    sample_dmp_1D_in                                = dlmread('../../data/dmp_1D/sample_traj_1.txt');
    sample_dmp_1D_goal                              = dlmread('goal_state_global_trajectory.txt');
    sample_dmp_1D_out                               = dlmread('transform_sys_state_global_trajectory.txt');
    sample_dmp_1D_canonical_sys_state_trajectory    = dlmread('canonical_sys_state_trajectory.txt');
    
    figure;
    hold on;
    grid on;
    if (is_plot_goal_trajectory == 0)
        if (is_plot_canonical_sys == 0)
            plot(sample_dmp_1D_in(:,1),sample_dmp_1D_in(:,2),...
                 sample_dmp_1D_out(:,1),sample_dmp_1D_out(:,2));
            legend('training data/trajectory','DMP-reproduced trajectory');
        else
            plot(sample_dmp_1D_in(:,1),sample_dmp_1D_in(:,2),...
                 sample_dmp_1D_out(:,1),sample_dmp_1D_out(:,2),...
                 sample_dmp_1D_canonical_sys_state_trajectory(:,1),sample_dmp_1D_canonical_sys_state_trajectory(:,2),...
                 sample_dmp_1D_canonical_sys_state_trajectory(:,1),sample_dmp_1D_canonical_sys_state_trajectory(:,3));
            legend('training data/trajectory','DMP-reproduced trajectory','canonical position','canonical velocity');
        end
    else
        if (is_plot_canonical_sys == 0)
            plot(sample_dmp_1D_in(:,1),sample_dmp_1D_in(:,2),...
                 sample_dmp_1D_goal(:,1),sample_dmp_1D_goal(:,2),...
                 sample_dmp_1D_out(:,1),sample_dmp_1D_out(:,2));
            legend('training data/trajectory','goal trajectory','DMP-reproduced trajectory');
        else
            plot(sample_dmp_1D_in(:,1),sample_dmp_1D_in(:,2),...
                 sample_dmp_1D_goal(:,1),sample_dmp_1D_goal(:,2),...
                 sample_dmp_1D_out(:,1),sample_dmp_1D_out(:,2),...
                 sample_dmp_1D_canonical_sys_state_trajectory(:,1),sample_dmp_1D_canonical_sys_state_trajectory(:,2),...
                 sample_dmp_1D_canonical_sys_state_trajectory(:,1),sample_dmp_1D_canonical_sys_state_trajectory(:,3));
            legend('training data/trajectory','goal trajectory','DMP-reproduced trajectory','canonical position','canonical velocity');
        end
    end
    title('Plot of DMP Trajectories');
    xlabel('time');
    ylabel('position');
    hold off;
    
    if (is_plot_forcing_term ~= 0)
        sample_dmp_1D_forcing_term_trajectory           = dlmread('forcing_term_trajectory.txt');
    
        figure;
        hold on;
        grid on;
        if (is_plot_forcing_term_vs_canonical_state ~= 0)
            plot(sample_dmp_1D_canonical_sys_state_trajectory(:,1),50*sample_dmp_1D_canonical_sys_state_trajectory(:,2),...
                 sample_dmp_1D_canonical_sys_state_trajectory(:,1),50*sample_dmp_1D_canonical_sys_state_trajectory(:,3),...
                 sample_dmp_1D_forcing_term_trajectory(:,1),sample_dmp_1D_forcing_term_trajectory(:,2));
            legend('50*canonical position','50*canonical velocity','forcing term');
            title('Plot of Forcing Term and Canonical System Trajectory');
            ylabel('canonical state variable or forcing term magnitude');
        else
            plot(sample_dmp_1D_forcing_term_trajectory(:,1),sample_dmp_1D_forcing_term_trajectory(:,2));
            legend('forcing term');
            title('Plot of Forcing Term Trajectory');
            ylabel('forcing term magnitude');
        end
        xlabel('time');
        hold off;
    end
end
