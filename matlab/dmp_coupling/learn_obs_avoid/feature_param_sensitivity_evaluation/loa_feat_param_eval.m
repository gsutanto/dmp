close                       all;
clear                       all;
clc;

addpath('../data/');
addpath('../utilities/');

D                           = 3;    % Number of dimensions

n_rfs                       = 25;   % Number of basis functions used to represent the forcing term of DMP
c_order                     = 1;    % DMP is using 2nd order canonical system

is_adding_random_variability= 0;

is_using_gsutanto_data      = 1;
is_using_akshara_data       = 0;

% feature method
loa_feat_method             = 0;
% loa_feat_method == 0: Akshara's feature
% loa_feat_method == 1: Kernelized "General" Features

is_loading_precomputed_DMP_w= 0;

Wn                          = 0.02;

is_plot_primitives          = 0;
is_plot_synth_obs_avoid_traj= 1;

flag                        = 'master_arm';
object                      = 'sph'; % 'ell' , 'cyl'

if (strcmp(flag, 'master_arm') == 1)
    % MasterArm robot's sampling rate is 420.0 Hz
    dt      = 1.0/420.0;
else
    if(strcmp(object,'sph'))
        % sphere data is sampled at 1000 Hz
        dt  = 0.001;
    else
        % ellipsoid and cylinder sampled at 100Hz
        dt  = 0.01;
    end
end

freq                        = 1/dt;

%% Obstacle Avoidance Features and Params Grid Setting

% Weights grid:
start_w_grid                    = log10(0.001);
end_w_grid                      = log10(1000);
n_w_grid                        = 1 + end_w_grid - start_w_grid;
loa_feat_param.w                = logspace(start_w_grid,end_w_grid,n_w_grid);

% Akshara-Franzi (Humanoids'14) features:
N_AF_H14_beta_phi1_phi2_grid	= 5;
AF_H14_beta_phi1_phi2_low       = 10.0/pi; % turns out too low of beta is not good, it doesn't converge to goal
AF_H14_beta_phi1_phi2_high      = 20.0/pi;
N_AF_H14_k_phi1_phi2_grid       = 1;
AF_H14_k_phi1_phi2_low          = 1.0;
AF_H14_k_phi1_phi2_high         = 5.0;

loa_feat_param     = initializeLearnObsAvoidFeatParam();
[ loa_feat_param ] = initializeAF_H14LearnObsAvoidFeatParam( loa_feat_param, ...
                                                             N_AF_H14_beta_phi1_phi2_grid, ...
                                                             AF_H14_beta_phi1_phi2_low, ...
                                                             AF_H14_beta_phi1_phi2_high, ...
                                                             N_AF_H14_k_phi1_phi2_grid, ...
                                                             AF_H14_k_phi1_phi2_low, ...
                                                             AF_H14_k_phi1_phi2_high );

% end of Obstacle Avoidance Features and Params Grid Setting

%% Load the Data

% Use the following if filtering is needed:
[b,a]                   = butter(2, Wn);

data_file   = {'data_multi_demo_static_preprocessed_02_2.mat','data_sph_new.mat'};

data        = cell(0,0);
for i=1:size(data_file,2)
    if (((is_using_gsutanto_data) && (i==1)) || ...
        ((is_using_akshara_data) && (i==2)))
        data_temp   = load(data_file{1,i});
        data        = [data; data_temp.data];
    end
end

global_buffer   = cell(size(data,1),10);

% Compute velocity and acceleration and append to the data:
for i=1:1
    
    % Baseline (without Obstacle) Behavior
    global_buffer{i,5}  = zeros(1,size(data{i,1},2));   % Baseline taus
    global_buffer{i,6}  = zeros(1,size(data{i,1},2));   % Baseline dts
    for j=1:size(data{i,1},2)
        % Baseline (without Obstacle) Demonstrations
        global_buffer{i,1}{1,j} = data{i,1}{1,j};
        global_buffer{i,1}{2,j} = data{i,1}{2,j};
        global_buffer{i,1}{3,j} = data{i,1}{3,j};
        
        global_buffer{i,3}      = cell(4,1);
        
        % Baseline (without Obstacle) taus and dts
        global_buffer{i,5}(1,j) = (size(global_buffer{i,1}{1,j},1)-1)*dt;
        global_buffer{i,6}(1,j) = dt;
    end
end

% end of Load the Data

%% Learning Baseline Primitive

if (is_loading_precomputed_DMP_w == 0)
    
    disp(['Learning Baseline Primitive']);

    for i=1:1
        disp(['    ', num2str(i)]);
        [ wi, Yifit, Ydifit, Yddifit, Fifit, mean_start, mean_goal, mean_dt ] = learnPrimitiveMulti( global_buffer{i,1}(1,:), global_buffer{i,1}(2,:), global_buffer{i,1}(3,:), global_buffer{i,5}, global_buffer{i,6}, n_rfs, c_order );
        mean_traj_length    = size(Fifit,1);
    end

    if (is_plot_primitives)
        figure;
        subplot(2,2,1);
            hold on;
            axis equal;
                plot3(Yifit(:,1),Yifit(:,2),Yifit(:,3),'r');
                for j=1:size(global_buffer{1,1},2)
                    plot3(global_buffer{1,1}{1,j}(:,1),global_buffer{1,1}{1,j}(:,2),global_buffer{1,1}{1,j}(:,3),'c:');
                end
                title('Yi: data vs fit');
                legend('fit');
            hold off;
        subplot(2,2,2);
            hold on;
            axis equal;
                plot3(Ydifit(:,1),Ydifit(:,2),Ydifit(:,3),'r');
                for j=1:size(global_buffer{1,1},2)
                    plot3(global_buffer{1,1}{2,j}(:,1),global_buffer{1,1}{2,j}(:,2),global_buffer{1,1}{2,j}(:,3),'c:');
                end
                title('Ydi: data vs fit');
                legend('fit');
            hold off;
        subplot(2,2,3);
            hold on;
            axis equal;
                plot3(Yddifit(:,1),Yddifit(:,2),Yddifit(:,3),'r');
                for j=1:size(global_buffer{1,1},2)
                    plot3(global_buffer{1,1}{3,j}(:,1),global_buffer{1,1}{3,j}(:,2),global_buffer{1,1}{3,j}(:,3),'c:');
                end
                title('Yddi: data vs fit');
                legend('fit');
            hold off;
        subplot(2,2,4);
            hold on;
            axis equal;
                plot3(Fifit(:,1),Fifit(:,2),Fifit(:,3),'r');
                title('Fi fit');
            hold off;
    end
    
    save('wi.mat', 'wi', 'mean_start', 'mean_goal', 'mean_dt', 'mean_traj_length');
else
    load('wi.mat');
end

% end of Learning Baseline Primitive

%% Unrolling Obstacle Avoidance Trajectories under Different Parameter Variations

for i=1:1

    global_buffer{i,2}      = data{i,2};                        % Obstacle Center Coordinate

    if (is_adding_random_variability)
        global_buffer{i,4}  = (1.0+abs(0.5*rand))*data{i,4};    % Obstacle Radius

        % Start and Goal Position of the Demonstrated Obstacle
        % Avoidance Behavior
        global_buffer{i,9}{1,1}     = mean_start + (0.1*rand(size(mean_start))); 
        global_buffer{i,10}{1,1}    = mean_goal + (0.1*rand(size(mean_goal)));
        
        % Vary the tau (movement duration) by multiplying traj_length with a random number:
        traj_length_i               = round((1.0+abs(rand))*mean_traj_length);
    else
        global_buffer{i,4}  = data{i,4};   % Obstacle Radius

        % Start and Goal Position of the Demonstrated Obstacle
        % Avoidance Behavior
        global_buffer{i,9}{1,1}     = mean_start; 
        global_buffer{i,10}{1,1}    = mean_goal;
        
        traj_length_i               = mean_traj_length;
    end

    % Obstacle Avoidance Behavior
    [ global_buffer{i,3}{1,1}, global_buffer{i,3}{2,1}, global_buffer{i,3}{3,1}, global_buffer{i,3}{4,1} ] = evaluateObsAvoidSphereUnderParam( wi, global_buffer{i,9}{1,1}, global_buffer{i,10}{1,1}, global_buffer{i,2}', traj_length_i, mean_dt, c_order, global_buffer{i,4}, loa_feat_method, loa_feat_param );
    
    % Obstacle Avoidance tau and dt
    global_buffer{i,7}(1,1) = (size(global_buffer{i,3}{1,1},1)-1)*dt;
    global_buffer{i,8}(1,1) = dt;
end

w_idx_1 = 5;
w_idx_2 = 6;
w_idx_3 = 7;

if (is_plot_synth_obs_avoid_traj)
    figure;
    hold on;
    axis equal;
    plot3(Yifit(:,1),Yifit(:,2),Yifit(:,3),'b');
    for t_idx=1:size(global_buffer{1,3}{1,1},1)
        plot3(global_buffer{1,3}{1,1}{t_idx,w_idx_1}(:,1),...
              global_buffer{1,3}{1,1}{t_idx,w_idx_1}(:,2),...
              global_buffer{1,3}{1,1}{t_idx,w_idx_1}(:,3),'c');
        plot3(global_buffer{1,3}{1,1}{t_idx,w_idx_2}(:,1),...
              global_buffer{1,3}{1,1}{t_idx,w_idx_2}(:,2),...
              global_buffer{1,3}{1,1}{t_idx,w_idx_2}(:,3),'g');
        plot3(global_buffer{1,3}{1,1}{t_idx,w_idx_3}(:,1),...
              global_buffer{1,3}{1,1}{t_idx,w_idx_3}(:,2),...
              global_buffer{1,3}{1,1}{t_idx,w_idx_3}(:,3),'r');
    end
    plot_sphere(global_buffer{1,4}, global_buffer{1,2}(1,1), global_buffer{1,2}(1,2), global_buffer{1,2}(1,3));
    for s_idx=1:size(data{1,3},2)
        plot3(data{1,3}{1,s_idx}(:,1),...
              data{1,3}{1,s_idx}(:,2),...
              data{1,3}{1,s_idx}(:,3),'k');
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
    legend('baseline traj');
    title('baseline vs obst avoid traj');
    hold off;
end

% end of Unrolling Obstacle Avoidance Trajectories under Different Parameter Variations
