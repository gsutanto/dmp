close all;

addpath('../utilities/');

load('4_w_ard.mat');
load('4_loa_feat_param.mat');

goal_point  = [0.0; 0.0; 0.0];
obs_point   = [2.5; 2.5; 0.0];

x_values    = linspace(0.0, 5.0, 251);
y_values    = linspace(0.0, 5.0, 251);

[x,y]       = meshgrid(x_values, y_values);
pf_value    = zeros(size(x,1),size(x,2),size(loa_feat_param.PF_DYN3_s_depth_grid,2));

% function to compute velocity directed to goal_point:
f_vel      	= @(t,Y) [(goal_point-Y)/((norm(goal_point-Y)*(norm(goal_point-Y)~=0))+((norm(goal_point-Y)==0)))];

t           = 0;
tau         = 1.0;
for i = 1:size(x,1)
    for j = 1:size(x,2)
        disp(['Evaluating grid idx (',num2str(i),',',num2str(j),')']);
        endeff_position     = [x(i,j); y(i,j); 0.0];
        endeff_velocity     = f_vel(t, endeff_position);
        endeff_state{1,1}   = endeff_position;
        endeff_state{2,1}   = endeff_velocity;
        endeff_state{3,1}   = zeros(3,1);
        
        obs_state{1,1}   = obs_point;
        obs_state{2,1}   = zeros(3,1);
        obs_state{3,1}   = zeros(3,1);
        
        loa_pf_dyn3_value_candidates_per_point = w_ard.' .* computePF_DYN3ObstAvoidPotentialFuncValueFeatPerPoint( loa_feat_param, endeff_state, obs_state, tau );
        for s_idx = 1:size(loa_feat_param.PF_DYN3_s_depth_grid,2)
            involved_pf_dyn3_candidate_idx  = find(loa_feat_param.PF_DYN3_s_rowcoldepth_vector == loa_feat_param.PF_DYN3_s_depth_grid(1,s_idx));
            if (length(involved_pf_dyn3_candidate_idx) ~= length(loa_feat_param.PF_DYN3_s_rowcoldepth_vector)/length(loa_feat_param.PF_DYN3_s_depth_grid))
                disp('WARNING: loa_feat_param is inconsistent!!!');
            end
            pf_value(i,j,s_idx) = sum(loa_pf_dyn3_value_candidates_per_point(:,involved_pf_dyn3_candidate_idx));
        end
    end
end

if (exist('./plot_potential_func_landscape_single_obs_point/', 'dir') ~= 7)  % if directory NOT exist
    mkdir('.', 'plot_potential_func_landscape_single_obs_point');	% create directory
end
                    
for s_idx = 1:size(loa_feat_param.PF_DYN3_s_depth_grid,2)
    h_fig   = figure;
    axis tight equal;
    hold on;
        h_surf{s_idx}   = surf(x,y,pf_value(:,:,s_idx));
        set(h_surf{s_idx},'LineStyle','none');
        figure_name     = ['U_kdyn3_s_',num2str(loa_feat_param.PF_DYN3_s_depth_grid(1,s_idx)),'.fig'];
        title(['U\_kdyn3 Surface Plot, s=',num2str(loa_feat_param.PF_DYN3_s_depth_grid(1,s_idx))]);
        xlabel('x');
        ylabel('y');
        zlabel('U\_kdyn3');
    hold off;
    
    savefig(h_fig, ['./plot_potential_func_landscape_single_obs_point/', figure_name]);
%     close(h_fig);
end