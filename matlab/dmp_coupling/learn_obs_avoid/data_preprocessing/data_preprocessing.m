close all;
clear all;
clc;

addpath('../data/');
addpath('../utilities/');

is_selecting_data_manually  = 0;

is_plot_processed_data      = 1;

data_option                 = 1;

D                           = 3;

Wn                          = 0.02;

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

data_tmp    = load('data_multi_demo_static_preprocessed_01.mat');
input_data  = data_tmp.data;
output_data = cell(size(input_data));

exclude_idx = cell(size(input_data,1),1);
retain_idx  = cell(size(input_data,1),1);

if (data_option == 1)
    exclude_idx{1,1}    = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,22,26,27,28,32];
%     exclude_idx{1,1}    = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,22,25,26,27,28,30,32];
%     exclude_idx{1,1}    = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22,23,25,26,27,28,29,30,31,32];
    exclude_idx{2,1}    = [1,2,4,5,6,7,8,9,10,11];
    exclude_idx{3,1}    = [1,2,3,6,7,8,9,11,12,13];
    exclude_idx{4,1}    = [1,2,5,6,7,8,10,12];
    exclude_idx{5,1}    = [1,2,3,7,9,10,11,12];
    exclude_idx{6,1}    = [1,14];
    exclude_idx{7,1}    = [1,2,3,4,5,7,8,10,11,12,13];
elseif (data_option == 2)
    exclude_idx{1,1}    = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,27,29,30,32];
    exclude_idx{2,1}    = [1,2,3,4,5,6,7,8,9,10,11,12];
    exclude_idx{3,1}    = [1,2,3,6,7,8,9,10,11,12,13,14];
    exclude_idx{4,1}    = [1,2,3,4,5,6,7,8,10,11,12];
    exclude_idx{5,1}    = [1,2,3,4,6,7,9,10,11,12];
    exclude_idx{6,1}    = [1,2,3,4,8,9,10,11,12,13,14,16,17,18];
    exclude_idx{7,1}    = [1,2,3,4,5,7,8,9,10,11,12,13,15];
end

% Use the following if filtering is needed:
[b,a]       = butter(2, Wn);

for i=1:size(input_data,1)
    % Baseline
    for j=1:size(input_data{i,1},2)
        output_data{i,1}{1,j}   = input_data{i,1}{1,j};
        output_data{i,1}{2,j}   = input_data{i,1}{2,j};
        output_data{i,1}{3,j}   = input_data{i,1}{3,j};
    end
    
    % Obstacle Description
    output_data{i,2}    = input_data{i,2};    % Obstacle Center Coordinate
    output_data{i,4}    = input_data{i,4};    % Obstacle Radius

    % Demonstrated Obstacle Avoidance
    for j=1:size(input_data{i,3},2)
        if (~(ismember(j,exclude_idx{i,1})))
            retain_idx{i,1} = [retain_idx{i,1}, j];
            tautmp  = (size(input_data{i,3}{1,j},1)-1)*dt;
            traj    = completeTrajectory(input_data{i,3}{1,j}, dt, tautmp);
            for d=1:D
                output_data{i,3}{1,size(retain_idx{i,1},2)}(:,d)    = filtfilt(b,a,traj(:,d));
                output_data{i,3}{2,size(retain_idx{i,1},2)}(:,d)    = diffnc(output_data{i,3}{1,size(retain_idx{i,1},2)}(:,d),dt);
                output_data{i,3}{3,size(retain_idx{i,1},2)}(:,d)    = diffnc(output_data{i,3}{2,size(retain_idx{i,1},2)}(:,d),dt);
            end
        end
    end
end

if (is_selecting_data_manually)
    for i=1:size(output_data,1)
        figure;
        for d=1:D
            subplot(D,1,d);
            hold on;
                for j=1:size(output_data{i,3},2)
                    plot(output_data{i,3}{1,j}(:,d),'b');
                    retain_idx{i,1}(1,j)
                    drawnow;
                end
                title_string    = ['Processed Yo dim ',num2str(d)];
                title(title_string);
            hold off;
        end

        figure;
        for d=1:D
            subplot(D,1,d);
            hold on;
                for j=1:size(output_data{i,3},2)
                    plot(output_data{i,3}{2,j}(:,d),'b');
                    retain_idx{i,1}(1,j)
                    drawnow;
                end
                title_string    = ['Processed Ydo dim ',num2str(d)];
                title(title_string);
            hold off;
        end

        figure;
        for d=1:D
            subplot(D,1,d);
            hold on;
                for j=1:size(output_data{i,3},2)
                    plot(output_data{i,3}{3,j}(:,d),'b');
                    retain_idx{i,1}(1,j)
                    drawnow;
                end
                title_string    = ['Processed Yddo dim ',num2str(d)];
                title(title_string);
            hold off;
        end
        
        keyboard;
    end
end

if (is_plot_processed_data)
    for i=1:size(output_data,1)
        figure;
        subplot(2,2,1);
            hold on;
            axis equal;
                for j=1:size(output_data{i,3},2)
                    plot3(output_data{i,3}{1,j}(:,1),...
                          output_data{i,3}{1,j}(:,2),...
                          output_data{i,3}{1,j}(:,3),'b');
                    plot_sphere(output_data{i,4}, ...
                                output_data{i,2}(1,1), ...
                                output_data{i,2}(1,2), ...
                                output_data{i,2}(1,3));
                    drawnow;
                end
                title_string    = ['Processed Yo Setting # ',num2str(i)];
                title(title_string);
            hold off;
        subplot(2,2,2);
            hold on;
            axis equal;
                for j=1:size(output_data{i,3},2)
                    plot3(output_data{i,3}{2,j}(:,1),...
                          output_data{i,3}{2,j}(:,2),...
                          output_data{i,3}{2,j}(:,3),'b');
                    drawnow;
                end
                title_string    = ['Processed Ydo dim ',num2str(d)];
                title(title_string);
            hold off;
        subplot(2,2,3);
            hold on;
            axis equal;
                for j=1:size(output_data{i,3},2)
                    plot3(output_data{i,3}{3,j}(:,1),...
                          output_data{i,3}{3,j}(:,2),...
                          output_data{i,3}{3,j}(:,3),'b');
                    drawnow;
                end
                title_string    = ['Processed Yddo dim ',num2str(d)];
                title(title_string);
            hold off;
    end
end

data        = output_data;

save_title  = ['../data/data_multi_demo_static_preprocessed_02_',num2str(data_option),'.mat'];
save(save_title, 'data');