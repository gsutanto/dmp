% Author : Giovanni Sutanto
% Date   : July 6, 2017

clear all;
close all;
clc;

prim_no         = 2;
ct_dim_no       = 5;
N_phase_display = 5;

addpath('../../../dmp_general/');

task_type                   = 'scraping';
load(['../dataset_Ct_tactile_asm_',task_type,'_augmented.mat']);
data_path   = ['../../../../data/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/pmnn/cpp_models/prim', ...
               num2str(prim_no), '/', num2str(ct_dim_no-1), '/'];

global      dcps;

n_rfs     	= 25;
c_order    	= 1;    % using 2nd order canonical system

assert(N_phase_display < n_rfs, 'N_phase_display must be smaller than n_rfs');

dcp_franzi('init', 1, n_rfs, num2str(1), c_order);
basis_centers 	= dcps.c;

ref_setting_no 	= 2;    % reference setting
ref_trial_no   	= 3;   % reference trial of reference setting

ref_phase_X   	= dataset_Ct_tactile_asm.sub_phase_X{prim_no, ref_setting_no}{ref_trial_no,1};
ref_normalized_phase_PSI_mult_phase_V   = dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{prim_no, ref_setting_no}{ref_trial_no,1};

ref_indices   	= zeros(size(basis_centers));

for nc=1:n_rfs
    distance_vector     = abs(ref_phase_X - basis_centers(nc,1));
    [~, ref_idx]        = min(distance_vector);
    ref_indices(nc,1)   = ref_idx;
end
ref_indices_indices     = round([1:N_phase_display].' * ((1.0 * n_rfs)/(N_phase_display+1)));

w1          = dlmread([data_path, '/w1']);
w2          = dlmread([data_path, '/w2']);

feature_priority_evolution  = zeros(size(w1,2), size(w1,1), N_phase_display);

for i=1:N_phase_display
    selected_ref_normalized_phase_PSI_mult_phase_V  = ref_normalized_phase_PSI_mult_phase_V(ref_indices(ref_indices_indices(i,1)),:);
    feature_priority_evolution(:,:,i)   = abs(w1 .* repmat((selected_ref_normalized_phase_PSI_mult_phase_V .* w2.'), size(w1, 1), 1)).';

    % Normalize the Priority
    % for p=1:n_rfs
    %     feature_priority    = feature_priority_evolution(p,:);
    %     if (all(feature_priority == 0))
    %         feature_priority= 0.0;
    %     else
    %         feature_priority= (1.0/(max(feature_priority) - min(feature_priority))) * (feature_priority - min(feature_priority));
    %     end
    %     feature_priority_evolution(p,:) = feature_priority;
    % end

    figure;
    imagesc(feature_priority_evolution(:,:,i));
    title(['prim #', num2str(prim_no), ', ct dim #', num2str(ct_dim_no), ': feature priority evolution over phase']);
    xlabel('features');
    ylabel('phase');
end

dominant_features_vs_phase  = abs(w1.');
for p=1:n_rfs
    features                    = dominant_features_vs_phase(p,:);
    sorted_descend_dom_feats    = sort(features, 2, 'descend');
    threshold                   = sorted_descend_dom_feats(1, 10);
    dominant_features           = zeros(size(features));
    dominant_features(1, find(features >= threshold))   = 1.0;
    dominant_features(1, find(features <  threshold))   = 0.0;
    dominant_features_vs_phase(p,:) = dominant_features;
end

figure;
imagesc(dominant_features_vs_phase);
title(['Top 10 Dominant Regular Hidden Layer Features for Each Phase RBF in Primitive #', num2str(prim_no), ', Roll-Orientation Coupling Term']);
xlabel('Regular Hidden Layer Features');
ylabel('Phase RBF');
set(gca, 'FontSize', 36);