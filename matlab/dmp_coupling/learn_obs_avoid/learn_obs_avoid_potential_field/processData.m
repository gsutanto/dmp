function [data_all, ct_all, fi_all, fot_all, w_all, ci, Di] = processData(data, dt, n_rfs)

% close all
addpath('util/');
addpath('trakSTAR/');
% Number of subjects
nsubj   = size(data,1);
% nsubj = 1;

debug=0;

%flag for using average forcing term for obstacle avoidance trajectories
use_avg_forcing_term=1;

count=1;

for ns=1:nsubj
    data_initial= data{ns,1};
    obs = data{ns,2};
    data_obstacle= data{ns,3};
    %obs=mean(obstacle);
    if(debug)
        for dim = 1:3
            figure,subplot(2,1,1), hold on
            for i = 1:length(data_initial)
                plot(data_initial{i}(:,dim))
            end
            hold off
            subplot(2,1,2), hold on
            for i = 1:length(data_obstacle)
                plot(data_obstacle{i}(:,dim))
            end
            hold off
        end
%       keyboard;
      close all
      data_disp = min(length(data_initial), length(data_obstacle));
      for i = 1 : data_disp
         figure(100), hold on
         plot3(data_initial{i}(:,1), data_initial{i}(:,2), data_initial{i}(:,3))
         plot3(data_obstacle{i}(:,1), data_obstacle{i}(:,2), data_obstacle{i}(:,3), 'r')
      end
%       keyboard
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
