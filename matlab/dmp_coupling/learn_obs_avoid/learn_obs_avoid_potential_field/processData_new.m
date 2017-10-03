function [data_all, ct_all, fi_all, fot_all, w_all, ci, Di] = ...
    processData_new(data, dt, n_rfs)

% close all
addpath('util/');
addpath('trakSTAR/');
% Number of subjects
nsubj=length(data)-1;

% DMP parameters
debug=0;

%flag for using average forcing term for obstacle avoidance trajectories
use_avg_forcing_term=1;
[b,a] = butter(2,0.1);

count=1;

for ns=1:nsubj
    clear data_initial data_obstacle
    %data_initial= base_data;
    obs = data{ns, 2};
    %data_obstacle= obs_data{ns};
    %obs=mean(obstacle);
    data_initial_old = data{ns, 1}; %filter_data(a,b,data{ns, 1});
    data_obstacle_old = data{ns, 3}; %filter_data(a,b,data{ns, 3});
    
    for i = 1:length(data_initial_old)
        data_initial{i} = post_process(data_initial_old{i});
    end
    
    obs_count  = 1;
    for i = 1:length(data_obstacle_old)
        if(length(data_obstacle_old{i}(:,1)) < 300)
            continue
        end
        data_obstacle{obs_count} = post_process(data_obstacle_old{i});
        obs_count = obs_count+1;
    end
    
    if(debug)
        for dim = 1:3
            figure(dim),clf,subplot(2,1,1), hold on
            for i = 1:length(data_initial_old)
                data_initial{i} = post_process(data_initial_old{i});
                plot(data_initial{i}(:,dim))
            end
            hold off
            subplot(2,1,2), hold on
            for i = 1:length(data_obstacle)
                plot(data_obstacle{i}(:,dim))
            end
            hold off
        end
      %keyboard;
      %close all
      
      data_disp = min(length(data_initial), length(data_obstacle));
      figure(100), clf, hold on
      for i = 1 : data_disp
         plot3(data_initial{i}(:,1), data_initial{i}(:,2), data_initial{i}(:,3))
         plot3(data_obstacle{i}(:,1), data_obstacle{i}(:,2), data_obstacle{i}(:,3), 'r')
      end
      keyboard
    end
    % compute coupling terms
    [ct, fi, fot, wi, wo, ci, Di] = computeCouplingTermsMulti(data_initial,data_obstacle,n_rfs,dt,use_avg_forcing_term);

    ct_all{count}=ct;
    data_all{count,1}=data_initial;
    data_all{count,2}=obs;
    data_all{count,3}=data_obstacle;
    fi_all{count,1} = fi;
    fot_all{count,1} = fot;
    w_all{count,1} = wo;
    w_all{count,2} = wi;
    count=count+1;
end
end



function [Ct, Fi, Fot, wi, wo, ci, Di] = computeCouplingTermsMulti(data_init,data_obs,n_rfs,dt,use_avg_forcing_term)

Tx = cell(length(data_init),1);
Ty = cell(length(data_init),1);
Tz = cell(length(data_init),1);

for i= 1:length(data_init)%
    
    xf = data_init{i}(:,1);
    yf = data_init{i}(:,2);
    zf = data_init{i}(:,3);
    
    % put start pos back to (0,0) - I used to move each trajectory such
    % that the starting position would be 0 - I don't remember exactly
    % why - might not be necessary here, so for now I leave it out
    %         xf = xf - xf(1);
    %         yf = yf - yf(1);
    %         zf = zf - zf(1);
    Tx{i} = xf;
    Ty{i} = yf;
    Tz{i} = zf;
end


Tox = cell(length(data_obs),1);
Toy = cell(length(data_obs),1);
Toz = cell(length(data_obs),1);


for i= 1:length(data_obs)%
    
    xf = data_obs{i}(:,1);
    yf = data_obs{i}(:,2);
    zf = data_obs{i}(:,3);
    
    % put start pos back to (0,0) - I used to move each trajectory such
    % that the starting position would be 0 - I don't remember exactly
    % why - might not be necessary here, so for now I leave it out
    %         xf = xf - xf(1);
    %         yf = yf - yf(1);
    %         zf = zf - zf(1);
    Tox{i} = xf;
    Toy{i} = yf;
    Toz{i} = zf;
end


[wix,Fx,Ftx,cx,Dx] = computeForcingTermMulti('batch_fit_multi',Tx,n_rfs,dt);
[wiy,Fy,Fty,cy,Dy] = computeForcingTermMulti('batch_fit_multi',Ty,n_rfs,dt);
[wiz,Fz,Ftz,cz,Dz] = computeForcingTermMulti('batch_fit_multi',Tz,n_rfs,dt);
wi = [wix, wiy, wiz];
ci = [cx, cy, cz];
Di = [Dx, Dy, Dz];

% 
% F = [Fx,Fy,Fz];
% Ft = [Ftx, Fty, Ftz];

if(use_avg_forcing_term)
    
    [Ctx,Fix,Fotx, wox] = computeCouplingTermMulti('batch_compute_coupling_term_avg',Tox,n_rfs,dt,wix);
    [Cty,Fiy,Foty, woy] = computeCouplingTermMulti('batch_compute_coupling_term_avg',Toy,n_rfs,dt,wiy);
    [Ctz,Fiz,Fotz, woz] = computeCouplingTermMulti('batch_compute_coupling_term_avg',Toz,n_rfs,dt,wiz);
    
    Ct = [Ctx, Cty, Ctz];
    Fi = [Fix, Fiy, Fiz];
    Fot = [Fotx, Foty, Fotz];
    wo = [wox, woy, woz];
    
else
    [Ctx,Fix,Fotx] = computeCouplingTermMulti('batch_compute_coupling_term',Tox,n_rfs,dt,wix);
    [Cty,Fiy,Foty] = computeCouplingTermMulti('batch_compute_coupling_term',Toy,n_rfs,dt,wiy);
    [Ctz,Fiz,Fotz] = computeCouplingTermMulti('batch_compute_coupling_term',Toz,n_rfs,dt,wiz);

    Ct = [Ctx, Cty, Ctz];
    Fi = [Fix, Fiy, Fiz];
    Fot = [Fotx, Foty, Fotz];
end


end
function data = filter_data(a,b,data)

    for i = 1:size(data,2)
        tx = data{i};
        tf = tx';
        for j = 1:3
            x = tx(:,j);
            tf(j,:) = filtfilt(b,a,x');
        end
        data{i} = tf';
    end
end


function data = post_process(data)
start = mean(data(1:10,:));
goal = mean(data(end-10:end,:));
    
    ctraj = data;
    
    start_pos = 1;
    for i = 1:size(ctraj,1)
        if( norm(ctraj(i,:) - start) < 0.02 )
            start_pos = start_pos+1;
        end
    end
    
    start_pos = max(1,start_pos-100);
    ctraj = ctraj(start_pos:end,:);
    dctraj = diff(ctraj);
    end_pos = size(ctraj,1);
    for i = 10:size(dctraj,1)
        %         if( i < 700 && norm(var(dctraj( (i-9):i ,:))) < 1e-9 )
        %             start_pos = start_pos + 1;
        %         else
        if(i >= 800 && norm(var(dctraj( (i-9):i ,:))) < 1e-10)
            end_pos = i+9;
            break;
        end
    end
    
    
    
    end_pos = min(end_pos,size(ctraj,1));
    ctraj = ctraj(1:end_pos,:);
    
    % now add the starting/end point 50 times in the beginning/end of traj
    
    % this is a hack to get 0 acceleration/velocity in the end
    % doesn't work to well because it creates discontinuities
    % If this is a real issue (non-zero acc/vel at end of traj) then the
    % "right" thing to do is compute the acc traj, compute a spline to
    % connect the current end of traj to 0 acceleration and then double
    % integrate to get back the position trajectory
    %     ctraj2 = zeros(size(ctraj,1) +100, size(ctraj,2));
    %     ctraj2(1:50,:) = repmat(ctraj(1,:),[50,1]);
    %     ctraj2(51:(end-50),:) = ctraj(1:end,:);
    %     ctraj2((end-50+1):end,:) = repmat(ctraj(end,:),[50,1]);
    %     data{nd} = ctraj2;
    
    data = ctraj;
    clear ctraj;
end
