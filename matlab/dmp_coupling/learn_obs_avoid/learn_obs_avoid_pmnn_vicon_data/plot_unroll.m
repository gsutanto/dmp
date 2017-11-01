clear all;
close all;
clc;

addpath('../utilities/');

load('data_multi_demo_vicon_static_global_coord.mat');
load('dataset_Ct_obs_avoid.mat');
load('unroll_dataset_learned_Ct_obs_avoid.mat');
load('no_dynamics_unroll_dataset_learned_Ct_obs_avoid.mat');

subset_settings_indices     = [1:22:222];
subset_demos_indices        = [1:1];
mode_stack_dataset          = 2;
feature_type                = 'raw';
N_primitive                 = size(dataset_Ct_obs_avoid.sub_Ct_target, 1);

D                           = 3;

for np=1:N_primitive
    for ns=1:length(subset_settings_indices)
        is      = subset_settings_indices(ns);
        for nd=1:length(subset_demos_indices)
            id  = subset_demos_indices(nd);
            figure;
            for d=1:D
                subplot(D,1,d);
                hold on;
                    plot(dataset_Ct_obs_avoid.sub_Ct_target{np,is}{id,1}(:,d),'r');
                    plot(unroll_dataset_learned_Ct_obs_avoid.sub_Ct_target{np,is}{id,1}(:,d),'b');
                    plot(no_dynamics_unroll_dataset_learned_Ct_obs_avoid.sub_Ct_target{np,is}{id,1}(:,d),'g');
                    if (d==1)
                        title(['coupling term: target vs unroll vs unroll (no dynamics) for setting #',num2str(is),', demo #',num2str(id)]);
                        legend('target', 'unroll', 'unroll (no dynamics)');
                    elseif (d==3)
                        xlabel('time');
                    end
                    ylabel(['d=',num2str(d)]);
                hold off;
            end
        end
    end
end