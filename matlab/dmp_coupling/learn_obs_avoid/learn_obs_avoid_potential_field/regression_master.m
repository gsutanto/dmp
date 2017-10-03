% This is the master file for running regression over the different object
% types. Even though the features are the same, they needed to be calculated
% slightly differently due to differnt nearest points. 

% Giovanni's Note:
% This implementation is NOT exactly the same as the C++ implementation,
% in the following sense:
% (1) The DMP does NOT implement local coordinate transformation.
% (2) The obstacle avoidance coupling term is NOT computed in the local
%     coordinate system (similar as DMP), so reproducibility in a different
%     context (i.e. different start and goal of the trajectory) is NOT
%     guaranteed.

%So, we have differnt flags for different objects to calculate the
%features.
close all;
clear all;
clc;
%The rest of the calculations remain largely the same across obstacles.
addpath('util/');
addpath('trakSTAR/');

%%%%%%%%%%%%% Option Flags %%%%%%%%%%%%%
debugging                           = 0;

use_gsutanto_data                   = 1;
use_akshara_data                    = 0;

use_humanoid14_features             = 1;
use_lyapunov_features               = 0;
use_recorded_lyapunov_features      = 0;
use_switching_controller_features   = 0;

use_feature_variance_thresholding   = 0;

use_positivity_constraints          = 0;
perform_ard                         = 1;

grid_option                         = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Thresholds %%%%%%%%%%%%%%
% feature_variance_threshold is a threshold for (allowed minimum) feature variance:
feature_variance_threshold          = 1e-15;

% Good in use with positivity constraint:
% feature_variance_threshold          = 1e-2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

object = 'sph'; % 'ell' , 'cyl'

% Giovanni's data:
% data_file   = {'data_multi_demo_static_preprocessed.mat','data_sph_new.mat'};
flag        = 'master_arm';

% DMP parameters
% franzi: we are setting dt in several files - differently, makes it hard
% to debug, I will remove the possibility to learn across objects for now
% to simplify the debugging process
n_rfs = 25;
ND=3;

% setting DMP parameters here so that they are consistent through out
% the whole process
if (strcmp(flag, 'master_arm') == 1)
    % MasterArm robot's sampling rate is 420.0 Hz
    dt = 1/420.0;
else
    if(strcmp(object,'sph'))
        % sphere data is sampled at 1000 Hz
        dt=0.001;
    else
        % ellipsoid and cylinder sampled at 100Hz
        dt=0.01;
    end
end

% Akshara's data:
% data_file = ['data_' object '_new.mat'];
% flag = 'vicon';
% Akshara's data:
data_file   = {'data_multi_demo_static_raw.mat','data_sph_new.mat'};

data        = cell(0,0);
for i=1:size(data_file,2)
    if (((use_gsutanto_data) && (i==1)) || ...
        ((use_akshara_data) && (i==2)))
        data_temp   = load(data_file{1,i});
        data        = [data; data_temp.data];
    end
end
[data, ct, fi, fo, w, ci, Di] = processData(data, dt, n_rfs);
% [data, ct, fi, fo, w, ci, Di] = processData_new(data, dt, n_rfs);
data_sph = data;

% processing data to return vectors for each ssubject - do I need to do
% this if I am going to go to average?

nsubj = size(data,1);

% these datafiles will now be used in a regression function

% Hyperparameter grid for the coupling term features
if (grid_option == 0)
    % ORIGINAL:
    n_hum14_grid1a=2;
    n_hum14_grid1b=1;
    n_hum14_n1=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n2=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n3=1;
    
    n_lyap_grid1a=5;
    n_lyap_grid1b=5;
    n_lyap_n1=n_lyap_grid1a*n_lyap_grid1b;
    n_lyap_n2=26;
    n_lyap_n3=100;
elseif (grid_option == 1)
    % TEST GSUTANTO:
    n_hum14_grid1a=1;
    n_hum14_grid1b=100;
    n_hum14_n1=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n2=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n3=51;
    
    n_lyap_grid1a=10;
    n_lyap_grid1b=10;
    n_lyap_n1=n_lyap_grid1a*n_lyap_grid1b;
    n_lyap_n2=51;
    n_lyap_n3=100;
elseif (grid_option == 2)
    n_hum14_grid1a=5;
    n_hum14_grid1b=5;
    n_hum14_n1=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n2=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n3=5;
    
    n_lyap_grid1a=5;
    n_lyap_grid1b=5;
    n_lyap_n1=n_lyap_grid1a*n_lyap_grid1b;
    n_lyap_n2=13;
    n_lyap_n3=25;
 end

% if debuggiing, make only three featurees, to plot against target
if(debugging)
    n_hum14_grid1a=3;
    n_hum14_grid1b=25;
    n_hum14_n1=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n2=n_hum14_grid1b*n_hum14_grid1a;
    n_hum14_n3=26;
    
    n_lyap_grid1a=5;
    n_lyap_grid1b=5;
    n_lyap_n1=n_lyap_grid1a*n_lyap_grid1b;
    n_lyap_n2=26;
    n_lyap_n3=50;
end

if (grid_option == 0)
    % ORIGINAL:
    [be_hum14, ko_hum14]=meshgrid(linspace(3/pi,5/pi,n_hum14_grid1a),linspace(1,2,n_hum14_grid1b))
    be1_hum14=reshape(be_hum14, n_hum14_n1,1)
    ko1_hum14=reshape(ko_hum14, n_hum14_n2,1)
    ko3_hum14=linspace(1,2,n_hum14_n3)
    
    [be_lyap, ko_lyap]=meshgrid(linspace(0.1,10.0,n_lyap_grid1a),linspace(0.1,10.0,n_lyap_grid1b))
    be1_lyap=reshape(be_lyap, n_lyap_n1,1)
    ko1_lyap=reshape(ko_lyap, n_lyap_n1,1)
    be2_lyap=linspace(2.0,52.0,n_lyap_n2)
    ko3_lyap=linspace(1,2,n_lyap_n3)
elseif (grid_option == 1)
    % TEST GSUTANTO:
    [be_hum14, ko_hum14]=meshgrid(linspace(1.5/pi,2.5/pi,n_hum14_grid1a),linspace(0.001,1.0,n_hum14_grid1b))
    be1_hum14=reshape(be_hum14, n_hum14_n1,1)
    ko1_hum14=reshape(ko_hum14, n_hum14_n2,1)
    ko3_hum14=linspace(0.01,10.0,n_hum14_n3)
    
    [be_lyap, ko_lyap]=meshgrid(linspace(0.001,10.0,n_lyap_grid1a),linspace(0.001,10.0,n_lyap_grid1b))
    be1_lyap=reshape(be_lyap, n_lyap_n1,1)
    ko1_lyap=reshape(ko_lyap, n_lyap_n1,1)
    be2_lyap=linspace(1.0,51.0,n_lyap_n2)
    ko3_lyap=linspace(0.001,1.0,n_lyap_n3)
elseif (grid_option == 2)
    [be_hum14, ko_hum14]=meshgrid(linspace(1.0/pi,5.0/pi,n_hum14_grid1a),linspace(0.1,10.0,n_hum14_grid1b))
    be1_hum14=reshape(be_hum14, n_hum14_n1,1)
    ko1_hum14=reshape(ko_hum14, n_hum14_n2,1)
    ko3_hum14=linspace(0.1,10.0,n_hum14_n3)
    
    [be_lyap, ko_lyap]=meshgrid(linspace(0.001,2.5,n_lyap_grid1a),linspace(0.001,2.5,n_lyap_grid1b))
    be1_lyap=reshape(be_lyap, n_lyap_n1,1)
    ko1_lyap=reshape(ko_lyap, n_lyap_n1,1)
    be2_lyap=linspace(2.0,52.0,n_lyap_n2)
    ko3_lyap=linspace(0.001,1.0,n_lyap_n3)
end

% Calculating the features, Fo, Fi, etc for mean trajectories

mean_ct = [];
phi_hum14 = [];
phi_lyap = [];
Y_all = [];
Yd_all = [];
Fo = [];
Fi = [];
traj_lengths = [];

for ns = 1:nsubj
    fprintf('ns = %d\n', ns);
    
    %Obstacle avoidance trajectory
    data_obstacle = data{ns,3};
    
    % obstacle avoidance weights
    wo = w{ns,1};
    
    %initial weights
    wi = w{ns,2};
    
    %obstacle position
    obs3 = data{ns,2};
    
    nL=length(data_obstacle);
    
    %time samples of trajectories
    nsamps = zeros(nL,1);
    %time durations
    taus = zeros(nL,1);
    %goal points
    goals = zeros(nL,3);
    %initial points
    y0s = zeros(nL,3);
    
    for i=1:nL
        traj = data_obstacle{i};
        nsamps(i) = size(traj,1);
        taus(i) = nsamps(i)*dt;
        goals(i,:) = traj(end,:);
        y0s(i,:) = traj(1,:);
    end
   
    % Need to adjust dt if all objects together
    % franzi: I don't understand why this makes sense if you want to reduce
    % the # of data points for the sphere you do not change tau - you
    % change the dt and the number of samples
    mean_nsamps = round(mean(nsamps));

    mean_tau = mean_nsamps*dt;
    mean_goal = mean(goals,1);
    mean_y0 = mean(y0s,1);
    

    Y = zeros(mean_nsamps,ND);
    mean_ctn = zeros(mean_nsamps,ND);
    Yd = Y; Ydd = Y;
    Fon = Y; Fin = Y; Yi = Y; Yid = Y; Yidd = Y;
    %     for i=1:length(data_obstacle)
    
    for d=1:3
        [Y(:,d), Yd(:,d), Ydd(:,d),Fod]=computeTrajectory(n_rfs, mean_tau, dt,wo(:,d), mean_y0(d), mean_goal(d), mean_nsamps);
        [Yi(:,d), Yid(:,d), Yidd(:,d),Fid]=computeTrajectory(n_rfs, mean_tau, dt,wi(:,d), mean_y0(d), mean_goal(d), mean_nsamps);
        Fon(:,d) = Fod;
        Fin(:,d) = Fid;
        mean_ctn(:,d) = Fod - Fid;
    end
    
%     if(debugging)  
%         figure(100), clf,
%         title('The original trajectories')
%         subplot(3,1,1), hold on, plot(Y(:,1),'r'), plot(Yi(:,1));
%         subplot(3,1,2), hold on, plot(Y(:,2),'r'), plot(Yi(:,2));
%         subplot(3,1,3), hold on, plot(Y(:,3),'r'), plot(Yi(:,3));
%         keyboard
%         
%         figure(101), clf,
%         title('3D original trajectories')
%         plot3(Y(:,1),Y(:,2),Y(:,3),'r'), hold on
%         plot3(Yi(:,1), Yi(:,2), Yi(:,3))
%     end
    
    % Calculating features now
    if (use_humanoid14_features)
        phi_hum14_n = compute_features_humanoids14(Y,Yd,mean_tau,be1_hum14,ko1_hum14,ko3_hum14,obs3, object);
    end
    if (use_lyapunov_features)
        if (~use_recorded_lyapunov_features)
            phi_lyap_n  = compute_features_lyapunov_basis_functions(Y,Yd,mean_tau,be1_lyap,ko1_lyap,be2_lyap,ko3_lyap,obs3, object);
        end
    end
    if (use_switching_controller_features)
        Y_all   = [Y_all; Y];
        Yd_all  = [Yd_all; Yd];
    end
        
%     if(debugging)
%         figure(102), clf,
%         subplot(3,3,1), hold on, plot(phi_hum14_n(:,:,1))
%         subplot(3,3,2), hold on, plot(phi_hum14_n(:,:,2))
%         title('humanoid 14 features')
%         subplot(3,3,3), hold on, plot(phi_hum14_n(:,:,3))
%         subplot(3,3,4), hold on, plot(phi_lyap_n(:,:,1))
%         subplot(3,3,5), hold on, plot(phi_lyap_n(:,:,2))
%         title('lyapunov features')
%         subplot(3,3,6), hold on, plot(phi_lyap_n(:,:,3))
%         subplot(3,3,7), hold on, plot(mean_ctn(:,1),'b')
%         subplot(3,3,8), hold on, plot(mean_ctn(:,2),'b')
%         title('Ct target')
%         subplot(3,3,9), hold on, plot(mean_ctn(:,3),'b')
%         hold off;
%     end
    
    if (use_humanoid14_features)
        phi_hum14   = [phi_hum14; phi_hum14_n];
    end
    if (use_lyapunov_features)
        if (~use_recorded_lyapunov_features)
            phi_lyap    = [phi_lyap; phi_lyap_n];
        end
    end
    mean_ct         = [mean_ct; mean_ctn];
    Fo              = [Fo; Fon];
    Fi              = [Fi; Fin];
    traj_lengths    = [traj_lengths; mean_nsamps];
end

phi     = [];
if (use_humanoid14_features)
    phi = [phi, phi_hum14];
end
if (use_lyapunov_features)
    if (~use_recorded_lyapunov_features)
        keyboard;
    else
        load phi_lyap.mat
    end
    phi = [phi, phi_lyap];
end
if (use_switching_controller_features)
    k                   = [0.1, 0.5, 1.0, 5.0, 10.0];
%     k                   = [1.0];
    phi_sw_controllers  = compute_features_switching_controller(phi, Fi, Y_all, Yd_all, k, obs3, object);
    phi                 = [phi_sw_controllers];
end

% if(debugging)
%     figure(102), clf,
%     subplot(3,3,1), hold on, plot(phi_hum14(:,:,1))
%     subplot(3,3,2), hold on, plot(phi_hum14(:,:,2))
%     title('humanoid 14 features')
%     subplot(3,3,3), hold on, plot(phi_hum14(:,:,3))
%     subplot(3,3,4), hold on, plot(phi_lyap(:,:,1))
%     subplot(3,3,5), hold on, plot(phi_lyap(:,:,2))
%     title('lyapunov features')
%     subplot(3,3,6), hold on, plot(phi_lyap(:,:,3))
%     subplot(3,3,7), hold on, plot(mean_ct(:,1),'b')
%     subplot(3,3,8), hold on, plot(mean_ct(:,2),'b')
%     title('Ct target')
%     subplot(3,3,9), hold on, plot(mean_ct(:,3),'b')
%     hold off;
% end

% gsutanto_modification:
figure;
clf;
for i=1:3
    subplot(2, 2, i);
    hold on;
    
    plot(Fi(:,i));
    plot(Fo(:,i));
    plot(mean_ct(:,i));
    legend('Fi', 'Fo', 'mean\_ct');
    if (i==1)
        title('x');
    elseif (i==2)
        title('y');
    elseif (i==3)
        title('z');
    end
    
    hold off;
end
% keyboard;

%zero mean data
old_ct = mean_ct;
meany = mean(mean_ct)
% mean_ct = bsxfun(@minus,mean_ct,meany);
% stdmean = std(mean_ct);
% mean_ct = mean_ct*diag(1./stdmean);

%feature matrix
x = reshape(phi,[size(phi,1), size(phi,2)*size(phi,3)]);
x_old = x;

%Normalize features
% orig_x = x;
% meanx = mean(x,1);
% x = bsxfun(@minus, x, meanx);
% stdx = std(x,1);
% x = x*diag(1.0./stdx);
% minx = min(x, [], 1);
% maxx = max(x, [], 1);
% x = bsxfun(@minus, x, minx);
% x = bsxfun(@rdivide, x, maxx - minx);
% x = [x ones(size(x,1),1)];

% keyboard;

if (use_feature_variance_thresholding)
    retain_idx  = find(var(x,0,1)>feature_variance_threshold); % only retain weights corresponding to rich enough features (to avoid numerical instability)
    x_new       = x(:,retain_idx);
    x           = x_new;
end

xx = x'*x;
xc = x'*mean_ct;

fprintf('Performing Weights Optimization ...\n');
if (use_positivity_constraints)
    w = zeros(size(xc));
    for d=1:3
        w(:,d)=quadprog((xx),-2*xc(:,d),[],[],[],[],zeros(size(xc,1),1),Inf(size(xc,1),1));
    end
else
    % simple (uniformly) regularized regression
    reg = 1e-5;
    A = reg*eye(size(x,2));
    w = (A + xx)\xc;
end
cfit = x*w;
var_ct = var(mean_ct,1);

figure;
hold on;
for d = 1:3
    subplot(2,2,d);
    plot(w(:,d));
    title(['Weights (no ARD), Axis ', num2str(d)]);
end

mse_no_ard = mean( (cfit-mean_ct).^2 );
nmse_no_ard = mse_no_ard./var_ct

max_w_no_ard    = max(max(w));
min_w_no_ard    = min(min(w));
fprintf('max w value: %f\n', max_w_no_ard);
fprintf('min w value: %f\n', min_w_no_ard);

% save('w_non_ard.mat','w')

%plot all trajectories at once
figure,
title('regression for all subjects (non-ARD)')
subplot(2,3,1),hold on,plot(Fi(:,1),'.b'),plot(Fo(:,1),'.c'),title('forcing terms d=1'),hold off;
subplot(2,3,2),hold on,plot(Fi(:,2),'.b'),plot(Fo(:,2),'.c'),title('forcing terms d=2'),hold off;
subplot(2,3,3),hold on,plot(Fi(:,3),'.b'),plot(Fo(:,3),'.c'),title('forcing terms d=3'),hold off;

subplot(2,3,4),hold on,plot(mean_ct(:,1),'.g'),plot(cfit(:,1),'.r'),title('coupling term fit d=1'), hold off;
subplot(2,3,5),hold on,plot(mean_ct(:,2),'.g'),plot(cfit(:,2),'.r'),title('coupling term fit d=2'), hold off;
subplot(2,3,6),hold on,plot(mean_ct(:,3),'.g'),plot(cfit(:,3),'.r'),title('coupling term fit d=3'), hold off;

disp(['without ARD: mse=', num2str(mse_no_ard), ', nmse=', num2str(nmse_no_ard)]);

% load w_non_ard.mat
if ((~use_positivity_constraints) && (perform_ard))
    % load w_ard.mat
    w_ard = zeros(size(w));
    for d = 1:3
       display(['ARD dim: ', num2str(d)]);
        [w_ard_d,act_idx] = ARD(x,mean_ct(:,d), debugging);
        w_ard(act_idx,d) = w_ard_d;
    end

    % [max_w_ard, idx_max_w_ard] = max(w_ard,[],1);
    % [min_w_ard, idx_min_w_ard] = min(w_ard,[],1);
    % w_ard   = 0.0*w_ard;
    % for d = 1:3
    %     w_ard(idx_max_w_ard(1,d),d)  = max_w_ard(1,d);
    %     w_ard(idx_min_w_ard(1,d),d)  = min_w_ard(1,d);
    % end
    % 
    figure;
    hold on;
    for d = 1:3
        subplot(2,2,d);
        plot(w_ard(:,d));
        title(['Weights after ARD, Axis ', num2str(d)]);
    end
    hold off;
    % 
    cfit_ard = x*w_ard;
    % 
    mse_ard= mean( (cfit_ard-mean_ct).^2 );
    nmse_ard = mse_ard./var_ct;

    max_w_ard    = max(max(w_ard));
    min_w_ard    = min(min(w_ard));
    fprintf('max w_ard value: %f\n', max_w_ard);
    fprintf('min w_ard value: %f\n', min_w_ard);

    % comment: displaying normalized nmse to better jugde how good/bad the
    % results are. if we would predict just zero the nmse is 1.0 
    disp(['without ARD: mse=', num2str(mse_no_ard), ', nmse=', num2str(nmse_no_ard)]);
    disp(['with ARD:    mse=', num2str(mse_ard), ', nmse=', num2str(nmse_ard)]);

    % ct_old_r  = bsxfun(@plus, (bsxfun(@minus, x_old, meanx))*diag(1.0./stdx)*w_ard, meany);

    % keyboard;
    % save('w_ard.mat','w_ard')

    figure,
    title('regression for all subjects (with ARD)')
    subplot(2,3,1),hold on,plot(Fi(:,1),'.b'),plot(Fo(:,1),'.c'),title('forcing terms d=1'),hold off;
    subplot(2,3,2),hold on,plot(Fi(:,2),'.b'),plot(Fo(:,2),'.c'),title('forcing terms d=2'),hold off;
    subplot(2,3,3),hold on,plot(Fi(:,3),'.b'),plot(Fo(:,3),'.c'),title('forcing terms d=3'),hold off;
    
    subplot(2,3,4),hold on,plot(mean_ct(:,1),'.g'),plot(cfit_ard(:,1),'.r'),title('coupling term fit ARD d=1'), hold off;
    subplot(2,3,5),hold on,plot(mean_ct(:,2),'.g'),plot(cfit_ard(:,2),'.r'),title('coupling term fit ARD d=2'), hold off;
    subplot(2,3,6),hold on,plot(mean_ct(:,3),'.g'),plot(cfit_ard(:,3),'.r'),title('coupling term fit ARD d=3'), hold off;
end

keyboard
traj_end = 0;
%or per subject
if(debugging)
    for ns = 1:nsubj
        traj_start = traj_end+1;
        traj_end = traj_start + traj_lengths(ns)-1;

        Fin = Fi(traj_start:traj_end,:);
        Fon = Fo(traj_start:traj_end,:);
        cfit_n = cfit(traj_start:traj_end,:);
        cfit_ard_n = cfit_ard(traj_start:traj_end,:);
        mean_ct_n = mean_ct(traj_start:traj_end,:);

        figure, 
        title(['regression subject ', num2str(ns)])
        subplot(3,3,1),hold on,plot(Fin(:,1),'b'),plot(Fon(:,1),'c'),title('forcing terms d=1'),hold off;
        subplot(3,3,2),hold on,plot(Fin(:,2),'b'),plot(Fon(:,2),'c'),title('forcing terms d=2'),hold off;
        subplot(3,3,3),hold on,plot(Fin(:,3),'b'),plot(Fon(:,3),'c'),title('forcing terms d=3'),hold off;

        subplot(3,3,4),hold on,plot(mean_ct_n(:,1),'g'),plot(cfit_n(:,1),'r'),title('coupling term fit d=1'), hold off;
        subplot(3,3,5),hold on,plot(mean_ct_n(:,2),'g'),plot(cfit_n(:,2),'r'),title('coupling term fit d=2'), hold off;
        subplot(3,3,6),hold on,plot(mean_ct_n(:,3),'g'),plot(cfit_n(:,3),'r'),title('coupling term fit d=3'), hold off;

        subplot(3,3,7),hold on,plot(mean_ct_n(:,1),'g'),plot(cfit_ard_n(:,1),'r'),title('coupling term fit ard d=1'), hold off;
        subplot(3,3,8),hold on,plot(mean_ct_n(:,2),'g'),plot(cfit_ard_n(:,2),'r'),title('coupling term fit ard d=2'), hold off;
        subplot(3,3,9),hold on,plot(mean_ct_n(:,3),'g'),plot(cfit_ard_n(:,3),'r'),title('coupling term fit ard d=3'), hold off;
        pause;

    end
end